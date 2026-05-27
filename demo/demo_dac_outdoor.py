#!/usr/bin/env python
"""
Depth-Any-Camera demo script for inference different types of camera data on a single perspective trained model.
Model: DAC-Indoor
Test data source: Scannet++, Matterport3D, NYU
"""

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import cv2
import torch
import torch.cuda as tcuda
from PIL import Image

from dac.models.idisc_erp import IDiscERP
from dac.models.idisc import IDisc
from dac.models.idisc_equi import IDiscEqui
from dac.models.cnn_depth import CNNDepth
from dac.utils.visualization import save_file_ply, save_val_imgs_v2
from dac.utils.unproj_pcd import reconstruct_pcd, reconstruct_pcd_erp
from dac.utils.erp_geometry import erp_patch_to_cam_fast, cam_to_erp_patch_fast, fisheye_mei_to_erp
from dac.dataloders.dataset import resize_for_input
import torchvision.transforms.functional as TF

##################################################################################################################
######################## samples for the demo of kitti360(fisheye), and kitti(perspective) #######################
##################################################################################################################

SAMPLE_1 = {
    "dataset_name": "kitti360",
    "image_filename": "demo/input/kitti360_rgb.png",
    "annotation_filename_depth": "demo/input/kitti360_depth.png",
    "depth_scale": 256.0,
    "fishey_grid": "splits/kitti360/grid_fisheye_02.npy",
    "crop_wFoV": 180, # degree decided by origianl data fov + some buffer
    "fwd_sz": (700, 700), # the patch size input to the model
    "erp": False,
    "cam_params": {
        'dataset':'kitti360',
        "fx": 1.3363220825849971e+03,
        "fy": 1.3357883350012958e+03,
        "cx": 7.1694323510126321e+02,
        "cy": 7.0576498308221585e+02,
        "xi": 2.2134047507854890e+00,
        "k1": 1.6798235660113681e-02,
        "k2": 1.6548773243373522e+00,
        "p1": 4.2223943394772046e-04,
        "p2": 4.2462134260997584e-04,
        # "w": 1400,
        # "h": 1400,
        "camera_model": "MEI",
    }
}

SAMPLE_2 = {
    "dataset_name": "kitti",
    "image_filename": "demo/input/kitti_rgb.png",
    "annotation_filename_depth": "demo/input/kitti_depth.png",
    "depth_scale": 256.0,
    "fishey_grid": None,
    "crop_wFoV": 100, # degree decided by origianl data fov + some buffer
    "fwd_sz": (300, 1000), # the patch size input to the model
    "erp": False,
    "cam_params": {
        'dataset': 'kitti',
        'fx': 7.188560e02,
        'fy': 7.188560e02,
        'cx': 6.071928e02,
        'cy': 1.852157e02,
        # "w": 1242,
        # "h": 375,
        "camera_model": "PINHOLE",
    }
}

def demo_one_sample(model, model_name, device, sample, cano_sz, args: argparse.Namespace):
    #######################################################################
    ############# data prepare (A simple version dataloader) ##############
    #######################################################################
    
    image = np.asarray(
        Image.open(sample["image_filename"])
    )
    org_img_h, org_img_w = image.shape[:2]
    if sample["annotation_filename_depth"] is None:
        depth = np.zeros((org_img_h, org_img_w), dtype=np.float32)
    else:
        depth = (
            np.asarray(
                cv2.imread(sample["annotation_filename_depth"], cv2.IMREAD_ANYDEPTH)
            ).astype(np.float32)
            / sample["depth_scale"]
        )
    
    dataset_name = sample["dataset_name"]
    fwd_sz=sample["fwd_sz"]
    
    if not sample["erp"]:
        # convert depth from zbuffer to euclid. Skip kitti360 because we prepared the depth already euclidean.
        if dataset_name in ['nyu', 'kitti']:
            x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
            depth = depth * np.sqrt((x - sample["cam_params"]['cx'])**2 + (y - sample["cam_params"]['cy'])**2 + sample["cam_params"]['fx']**2) / sample["cam_params"]['fx']
            depth = depth.astype(np.float32)
        
        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0

        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > 0.01
                
        # Automatically calculate the erp crop size
        crop_width = int(cano_sz[0] * sample["crop_wFoV"] / 180)
        crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])
        
        # convert to ERP
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            crop_height, crop_width, cano_sz[0], cano_sz[0]*2, sample["cam_params"], roll, scale_fac=None
        )
        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])
            
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask = resize_for_input((image * 255.).astype(np.uint8), depth, fwd_sz, None, [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask)
    else:
        attn_mask = np.ones_like(depth)
        lat_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
        long_range = torch.tensor([-np.pi, np.pi], dtype=torch.float32)
        
        # resizing process to fwd_sz.
        to_cano_ratio = cano_sz[0] / image.shape[0]
        image, depth, pad, pred_scale_factor = resize_for_input(image, depth, fwd_sz, None, cano_sz, to_cano_ratio)


    # convert to tensor batch
    normalization_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    image = TF.normalize(TF.to_tensor(image), **normalization_stats)
    gt = TF.to_tensor(depth)
    mask = TF.to_tensor((depth > 0.01).astype(np.uint8))
    attn_mask = TF.to_tensor((attn_mask>0).astype(np.float32)) # the non-empty region after ERP conversion
    batch = {
        "image": image.unsqueeze(0),
        "gt": gt.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "attn_mask": attn_mask.unsqueeze(0),
        "lat_range": lat_range.unsqueeze(0),
        "long_range": long_range.unsqueeze(0),
        "info": {
            "pred_scale_factor": pred_scale_factor,
        },
    }
    
    #######################################################################
    ########################### model inference ###########################
    #######################################################################

    gt, mask, attn_mask, lat_range, long_range = batch["gt"].to(device), batch["mask"].to(device), batch['attn_mask'].to(device), batch["lat_range"].to(device), batch["long_range"].to(device)
    with torch.no_grad():
        if model_name == "IDiscERP":
            preds, _, _ = model(batch["image"].to(device), lat_range, long_range)
        else:
            preds, _, _ = model(batch["image"].to(device))
    preds *= pred_scale_factor
    
    #######################################################################
    ##################  Visualization and Output results  #################
    #######################################################################
    save_img_dir = os.path.join(args.out_dir)
    os.makedirs(save_img_dir, exist_ok=True)
    if 'attn_mask' in batch.keys():
        attn_mask = batch['attn_mask'][0]
    else:
        attn_mask = None

    # adjust vis_depth_max for outdoor datasets
    if dataset_name == 'kitti360':
        vis_depth_max = 40.0
        vis_arel_max = 0.3
    elif dataset_name == 'kitti':
        vis_depth_max = 80.0
        vis_arel_max = 0.8
    else:
        # default indoor visulization parameters
        vis_depth_max = 10.0
        vis_arel_max = 0.5

    rgb = save_val_imgs_v2(
        0,
        preds[0],
        batch["gt"][0],
        batch["image"][0],
        f'{dataset_name}_output.jpg',
        save_img_dir,
        active_mask=attn_mask,
        valid_depth_mask=batch["mask"][0],
        depth_max=vis_depth_max,
        arel_max=vis_arel_max
    )
    
    pred_depth = preds[0, 0].detach().cpu().numpy()
    # if args.save_pcd:
    pcd = reconstruct_pcd_erp(pred_depth, mask=(batch['attn_mask'][0][0]).numpy(), lat_range=batch['lat_range'][0], long_range=batch['long_range'][0])
    save_pcd_dir = os.path.join(args.out_dir)
    os.makedirs(os.path.join(save_pcd_dir), exist_ok=True)
    pc_file = os.path.join(save_pcd_dir, f'{dataset_name}_pcd.ply')
    pcd = pcd.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    save_file_ply(pcd, rgb, pc_file)

    ##########  Convert the ERP result back to camera space for visualization (No need for original ERP image)  ##########
    if not sample['erp']:                    
        if dataset_name == 'kitti360':
            out_h = int(org_img_h/2)
            out_w = int(org_img_w/2)
            grid_fisheye = np.load(sample["fishey_grid"])
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'kitti360'}
        elif dataset_name == 'scannetpp':
            """
                Currently work perfect with phi = 0. For larger phi, corners may have artifacts.
            """
            grid_fisheye = np.load(sample["fishey_grid"])
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            out_h = int(org_img_h/2)
            out_w = int(org_img_w/2)
            grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (out_w, out_h))
            grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
            cam_params={'dataset':'scannetpp'} # when grid table is available, no need for intrinsic parameters
        else:
            # set output size the same aspact ratio as raw image (no need to be same as fw_size)
            out_h = org_img_h
            out_w = org_img_w
            grid_fisheye = None
            cam_params = sample["cam_params"]
            
        # scale the full erp_size depth scaling factor is equivalent to resizing data (given same aspect ratio)
        erp_h = cano_sz[0]
        erp_h = erp_h * batch['info']['pred_scale_factor']
        if 'f_align_factor' in batch['info']:
            erp_h = erp_h / batch['info']['f_align_factor'][0].detach().cpu().numpy()
        img_out, depth_out, valid_mask, active_mask, depth_out_gt = erp_patch_to_cam_fast(
            batch["image"][0], preds[0].detach().cpu(), attn_mask, 0., 0., out_h=out_h, out_w=out_w, erp_h=erp_h, erp_w=erp_h*2, cam_params=cam_params, 
            fisheye_grid2ray=grid_fisheye, depth_erp_gt=batch["gt"][0].detach().cpu())
        rgb = save_val_imgs_v2(
            0,
            depth_out,
            depth_out_gt,
            img_out,
            f'{dataset_name}_output_remap.jpg',
            save_img_dir,
            active_mask=active_mask,
            depth_max=vis_depth_max,
            arel_max=vis_arel_max
            )        


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = eval(config["model_name"]).build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()
    cano_sz=config["cano_sz"] # the ERP size model was trained on

    
    samples = [SAMPLE_1, SAMPLE_2]
    for i, sample in enumerate(samples):
        print(f"demo for sample {i}: {sample['dataset_name']}")
        demo_one_sample(model, config["model_name"], device, sample, cano_sz, args)
    print("Demo finished")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--config-file", type=str, default="checkpoints/dac_swinl_outdoor.json")
    parser.add_argument("--model-file", type=str, default="checkpoints/dac_swinl_outdoor.pt")
    parser.add_argument("--out-dir", type=str, default='demo/output')
    # parser.add_argument("--save-pcd", action="store_true")

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    main(config, args)