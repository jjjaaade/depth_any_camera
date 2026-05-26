#!/usr/bin/env python
"""
Depth-Any-Camera demo script for inference different types of camera data on a single perspective trained model.
Model: DAC-Outdoor
Test data source: KITTI360 (fisheye) and KITTI (perspective)

Extra:
    - Folder inference on custom perspective/pinhole images (optionally undistort).
"""

import argparse
import glob
import json
import math
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
from dac.utils.visualization import save_file_ply, save_val_imgs_v2, save_val_imgs_metric_values
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

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_intrinsics_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        intr = json.load(f)
    if not isinstance(intr, dict):
        raise ValueError(f"Invalid intrinsics JSON: {path}")
    if "dataset" not in intr:
        intr["dataset"] = "custom"
    if "camera_model" not in intr:
        intr["camera_model"] = "PINHOLE"
    return intr


def _infer_crop_wfov_deg(cam_params: Dict[str, Any], img_w: int, default_deg: float = 100.0, margin_deg: float = 10.0) -> float:
    """
    Infer a reasonable horizontal crop FoV (in degrees) from fx and image width.
    """
    fx = cam_params.get("fx", None)
    if fx is None:
        return default_deg
    try:
        fx = float(fx)
    except Exception:
        return default_deg
    if fx <= 0:
        return default_deg
    hfov_rad = 2.0 * math.atan(float(img_w) / (2.0 * fx))
    hfov_deg = hfov_rad * 180.0 / math.pi
    return float(min(179.0, max(10.0, hfov_deg + margin_deg)))


def _undistort_pinhole_if_needed(image_rgb: np.ndarray, cam_params: Dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Optional undistortion for perspective/pinhole images that are not pre-undistorted.

    Uses OpenCV radial-tangential (radtan) model with coefficients k1,k2,p1,p2,k3.
    """
    if not args.undistort:
        return image_rgb, cam_params

    required = ("fx", "fy", "cx", "cy")
    for k in required:
        if k not in cam_params:
            raise ValueError(f"--undistort requires '{k}' in intrinsics JSON (missing: {k})")

    h, w = image_rgb.shape[:2]
    k1 = float(cam_params.get("k1", 0.0))
    k2 = float(cam_params.get("k2", 0.0))
    p1 = float(cam_params.get("p1", 0.0))
    p2 = float(cam_params.get("p2", 0.0))
    k3 = float(cam_params.get("k3", 0.0))
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

    k_mat = np.array(
        [
            [float(cam_params["fx"]), 0.0, float(cam_params["cx"])],
            [0.0, float(cam_params["fy"]), float(cam_params["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    new_k = k_mat
    if args.undistort_optimal:
        new_k, _ = cv2.getOptimalNewCameraMatrix(k_mat, dist, (w, h), alpha=args.undistort_alpha, newImgSize=(w, h))

    undistorted = cv2.undistort(image_rgb, k_mat, dist, None, new_k)

    updated = dict(cam_params)
    updated["fx"] = float(new_k[0, 0])
    updated["fy"] = float(new_k[1, 1])
    updated["cx"] = float(new_k[0, 2])
    updated["cy"] = float(new_k[1, 2])
    updated["camera_model"] = "PINHOLE"
    return undistorted, updated


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


def run_custom_folder(model, model_name: str, device, config: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Run DAC-Outdoor on a folder of perspective images and save metric depth maps.

    Notes:
        - Output uint16 PNG: depth_uint16 = depth[m] * args.depth_scale
        - Also supports optional undistortion for distorted pinhole images (OpenCV radtan).
    """
    if args.intrinsics is None:
        raise ValueError("--intrinsics is required for --input-dir mode")

    cano_sz = config["cano_sz"]  # ERP size model was trained on
    cam_params_base = _load_intrinsics_json(args.intrinsics)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"--input-dir not found: {input_dir}")

    depth_dir = os.path.join(args.out_dir, "depth_uint16")
    npy_dir = os.path.join(args.out_dir, "depth_npy")
    vis_dir = os.path.join(args.out_dir, "vis")
    _ensure_dir(depth_dir)
    if args.save_npy:
        _ensure_dir(npy_dir)
    if args.vis:
        _ensure_dir(vis_dir)

    if args.glob is not None:
        image_paths = sorted(glob.glob(os.path.join(input_dir, args.glob), recursive=bool(args.recursive)))
    else:
        image_paths = []
        for ext in SUPPORTED_IMAGE_EXTS:
            if args.recursive:
                image_paths.extend(glob.glob(os.path.join(input_dir, "**", f"*{ext}"), recursive=True))
            else:
                image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir} (glob={args.glob!r})")

    fwd_sz = tuple(args.fwd_sz)
    if len(fwd_sz) != 2:
        raise ValueError("--fwd-sz must be two ints: H W")

    if args.output_downscale <= 0:
        raise ValueError("--output-downscale must be > 0")

    print(f"Found {len(image_paths)} images in {input_dir}")
    for idx, image_path in enumerate(image_paths):
        image = np.asarray(Image.open(image_path))
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]

        org_img_h, org_img_w = image.shape[:2]
        image, cam_params = _undistort_pinhole_if_needed(image, cam_params_base, args)

        # Dummy "gt depth" only to reuse the existing ERP conversion pipeline.
        depth_dummy = np.ones((image.shape[0], image.shape[1], 1), dtype=np.float32)
        mask_valid_depth = np.ones_like(depth_dummy, dtype=np.float32)

        crop_wfov = args.crop_wfov
        if crop_wfov is None:
            crop_wfov = _infer_crop_wfov_deg(cam_params, org_img_w, default_deg=100.0, margin_deg=args.crop_wfov_margin)

        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0

        image_f = image.astype(np.float32) / 255.0

        crop_width = int(cano_sz[0] * crop_wfov / 180.0)
        crop_height = int(crop_width * fwd_sz[0] / fwd_sz[1])

        # convert to ERP patch
        image_erp, depth_erp, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image_f,
            depth_dummy,
            mask_valid_depth,
            theta,
            phi,
            crop_height,
            crop_width,
            cano_sz[0],
            cano_sz[0] * 2,
            cam_params,
            roll,
            scale_fac=None,
        )

        lat_range = torch.tensor([float(np.min(latitude)), float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])

        # resizing process to fwd_sz.
        image_erp, depth_erp, _, pred_scale_factor, attn_mask = resize_for_input(
            (image_erp * 255.0).astype(np.uint8),
            depth_erp,
            fwd_sz,
            None,
            [image_erp.shape[0], image_erp.shape[1]],
            1.0,
            padding_rgb=[0, 0, 0],
            mask=erp_mask,
        )

        normalization_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        image_t = TF.normalize(TF.to_tensor(image_erp), **normalization_stats)
        attn_mask_t = TF.to_tensor((attn_mask > 0).astype(np.float32))

        batch = {
            "image": image_t.unsqueeze(0),
            "gt": TF.to_tensor(depth_erp).unsqueeze(0),
            "mask": TF.to_tensor((depth_erp > 0.01).astype(np.uint8)).unsqueeze(0),
            "attn_mask": attn_mask_t.unsqueeze(0),
            "lat_range": lat_range.unsqueeze(0),
            "long_range": long_range.unsqueeze(0),
            "info": {"pred_scale_factor": pred_scale_factor},
        }

        with torch.no_grad():
            if model_name == "IDiscERP":
                preds, _, _ = model(batch["image"].to(device), batch["lat_range"].to(device), batch["long_range"].to(device))
            else:
                preds, _, _ = model(batch["image"].to(device))
        preds *= pred_scale_factor

        # convert ERP patch output back to camera image coordinates.
        erp_h = cano_sz[0] * batch["info"]["pred_scale_factor"]
        if "f_align_factor" in batch["info"]:
            erp_h = erp_h / batch["info"]["f_align_factor"][0].detach().cpu().numpy()

        out_h = int(org_img_h / args.output_downscale)
        out_w = int(org_img_w / args.output_downscale)
        img_out, depth_out, _, active_mask = erp_patch_to_cam_fast(
            batch["image"][0],
            preds[0].detach().cpu(),
            batch["attn_mask"][0],
            0.0,
            0.0,
            out_h=out_h,
            out_w=out_w,
            erp_h=erp_h,
            erp_w=erp_h * 2,
            cam_params=cam_params,
            fisheye_grid2ray=None,
            depth_erp_gt=None,
        )

        if args.mirror_input_tree:
            rel_path = os.path.relpath(image_path, input_dir)
            stem, in_ext = os.path.splitext(rel_path)
        else:
            stem, in_ext = os.path.splitext(os.path.basename(image_path))

        depth_m = depth_out.squeeze().numpy().astype(np.float32)
        depth_uint16 = np.clip(depth_m * float(args.depth_scale), 0, 65535).astype(np.uint16)
        depth_ext = args.depth_ext
        if not depth_ext.startswith("."):
            depth_ext = f".{depth_ext}"
        depth_path = os.path.join(depth_dir, f"{stem}{depth_ext}")
        _ensure_dir(os.path.dirname(depth_path))
        cv2.imwrite(depth_path, depth_uint16)

        if args.save_npy:
            npy_path = os.path.join(npy_dir, f"{stem}.npy")
            _ensure_dir(os.path.dirname(npy_path))
            np.save(npy_path, depth_m)

        if args.vis:
            vis_ext = args.vis_ext if args.vis_ext is not None else in_ext
            if not vis_ext:
                vis_ext = ".jpg"
            if not vis_ext.startswith("."):
                vis_ext = f".{vis_ext}"
            vis_path = os.path.join(vis_dir, f"{stem}{args.vis_suffix}{vis_ext}")
            _ensure_dir(os.path.dirname(vis_path))
            save_val_imgs_metric_values(
                depth_out,
                img_out,
                os.path.basename(vis_path),
                os.path.dirname(vis_path),
                active_mask=active_mask,
                depth_max=args.vis_depth_max,
            )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] saved {depth_path}")


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = eval(config["model_name"]).build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()
    cano_sz=config["cano_sz"] # the ERP size model was trained on

    if args.input_dir is not None:
        run_custom_folder(model, config["model_name"], device, config, args)
        return

    
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
    parser.add_argument("--input-dir", type=str, default=None, help="Run custom folder inference when set.")
    parser.add_argument("--intrinsics", type=str, default=None, help="Camera intrinsics JSON for --input-dir mode.")
    parser.add_argument("--glob", type=str, default=None, help="Optional glob pattern inside --input-dir (e.g. '*.jpg' or '**/*.jpg' with --recursive).")
    parser.add_argument("--recursive", action="store_true", help="Recursively search --input-dir for images (enables '**' in --glob).")
    parser.add_argument("--mirror-input-tree", action="store_true", help="Mirror input subfolders and filenames under --out-dir outputs.")
    parser.add_argument("--depth-ext", type=str, default="png", help="Depth image extension for uint16 saving (e.g. 'png' or 'tiff').")
    parser.add_argument("--vis-ext", type=str, default=None, help="Visualization image extension. Default: same as input extension.")
    parser.add_argument("--vis-suffix", type=str, default="_vis", help="Suffix appended to visualization filename before extension.")
    parser.add_argument("--fwd-sz", type=int, nargs=2, default=[576, 1024], metavar=("H", "W"), help="Model input patch size (H W).")
    parser.add_argument("--crop-wfov", type=float, default=None, help="Horizontal crop FoV in degrees. If omitted, inferred from fx and image width.")
    parser.add_argument("--crop-wfov-margin", type=float, default=10.0, help="Extra degrees added to inferred FoV.")
    parser.add_argument("--output-downscale", type=float, default=1.0, help="Downscale output depth resolution (e.g. 2 -> half-res).")
    parser.add_argument("--depth-scale", type=int, default=1000, help="Scale factor for uint16 depth PNG (depth[m] * depth_scale).")
    parser.add_argument("--save-npy", action="store_true", help="Also save float32 depth in meters as .npy.")
    parser.add_argument("--vis", action="store_true", help="Save RGB|depth visualization images.")
    parser.add_argument("--vis-depth-max", type=float, default=80.0, help="Visualization max depth (meters).")
    parser.add_argument("--undistort", action="store_true", help="Undistort input images using k1,k2,p1,p2,k3 from --intrinsics (OpenCV radtan).")
    parser.add_argument("--undistort-optimal", action="store_true", help="Use getOptimalNewCameraMatrix for undistortion.")
    parser.add_argument("--undistort-alpha", type=float, default=0.0, help="Alpha for getOptimalNewCameraMatrix (0=crop, 1=keep all pixels).")
    # parser.add_argument("--save-pcd", action="store_true")

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    if args.input_dir is not None and args.intrinsics is None:
        parser.error("--intrinsics is required when --input-dir is set")

    main(config, args)
