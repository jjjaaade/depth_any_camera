#!/usr/bin/env python
"""
Depth-Any-Camera custom outdoor demo script (custom pinhole images).

Pipeline:
  1) Load raw image(s)
  2) Undistort (OpenCV radtan) and crop with alpha=0 ROI (optional/auto when distortion is detected)
  3) Convert undistorted pinhole image -> ERP patch (model input)
  4) Run model inference on ERP patch
  5) Inverse map predicted depth ERP patch -> pinhole image for visualization/output
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.cuda as tcuda
import torchvision.transforms.functional as TF
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dac.dataloders.dataset import resize_for_input
from dac.utils.erp_geometry import cam_to_erp_patch_fast, erp_patch_to_cam_fast
from dac.utils.visualization import save_val_imgs_metric_values

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


def _get_radtan_distortion(cam_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    required = ("fx", "fy", "cx", "cy")
    for k in required:
        if k not in cam_params:
            raise ValueError(f"Missing '{k}' in intrinsics JSON")

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
    return k_mat, dist


def _has_nonzero_distortion(dist: np.ndarray, eps: float = 1e-12) -> bool:
    return bool(np.any(np.abs(dist.astype(np.float64)) > eps))


def undistort_image(
    image_bgr: np.ndarray,
    camera_mat: np.ndarray,
    dist_coeff: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Undistort an image using OpenCV radtan model.

    Uses `getOptimalNewCameraMatrix(..., alpha=0.0)` and additionally crops to the returned ROI
    so the output has no black borders and intrinsics are adjusted accordingly.

    Returns:
        img_undistort_bgr, optimal_intrinsic (cropped), roi_xywh in the original image.
    """
    h, w = image_bgr.shape[:2]
    optimal_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
        np.array(camera_mat, dtype=np.float32),
        np.array(dist_coeff, dtype=np.float32),
        (int(w), int(h)),
        alpha=0.0,
        newImgSize=(int(w), int(h)),
    )
    img_undistort_full = cv2.undistort(
        image_bgr,
        np.array(camera_mat, dtype=np.float32),
        np.array(dist_coeff, dtype=np.float32),
        None,
        optimal_intrinsic,
    )

    x, y, rw, rh = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    if rw > 0 and rh > 0:
        x0 = max(0, min(w, x))
        y0 = max(0, min(h, y))
        x1 = max(0, min(w, x0 + rw))
        y1 = max(0, min(h, y0 + rh))
        img_undistort = img_undistort_full[y0:y1, x0:x1]
        optimal_intrinsic = optimal_intrinsic.copy()
        optimal_intrinsic[0, 2] -= float(x0)
        optimal_intrinsic[1, 2] -= float(y0)
        roi_xywh = (x0, y0, x1 - x0, y1 - y0)
    else:
        img_undistort = img_undistort_full
        roi_xywh = (0, 0, w, h)

    return img_undistort, optimal_intrinsic.astype(np.float32), roi_xywh


def _infer_fov_deg_from_focal(focal_px: float, size_px: int) -> float:
    return 2.0 * math.degrees(math.atan(float(size_px) / (2.0 * float(focal_px))))


def _save_depth_overlay_bgr(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    active_mask: np.ndarray,
    out_path: str,
    *,
    depth_max: Optional[float] = None,
    overlay_alpha: float = 0.55,
) -> None:
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError("rgb_bgr must be HxWx3")
    if depth_m.shape[:2] != rgb_bgr.shape[:2]:
        raise ValueError("depth_m must match rgb size")

    valid = np.isfinite(depth_m) & (depth_m > 0)
    if active_mask is not None:
        valid = valid & (active_mask > 0)
    if not np.any(valid):
        cv2.imwrite(out_path, rgb_bgr)
        return

    if depth_max is None:
        depth_max = float(np.percentile(depth_m[valid], 99))
    depth_max = max(1e-6, float(depth_max))

    depth_vis = depth_m.copy()
    depth_vis[~valid] = 0.0
    depth_u8 = np.clip(depth_vis / depth_max * 255.0, 0.0, 255.0).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
    heat_bgr[~valid] = 0

    blended = rgb_bgr.copy()
    blended[valid] = (rgb_bgr[valid] * (1.0 - overlay_alpha) + heat_bgr[valid] * overlay_alpha).astype(np.uint8)
    cv2.imwrite(out_path, blended)


def _infer_crop_wfov_deg(cam_params: Dict[str, Any], img_w: int, default_deg: float = 100.0, margin_deg: float = 10.0) -> float:
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


def _undistort_pinhole(
    image_rgb: np.ndarray,
    cam_params: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Undistort (OpenCV radtan) and, when `--undistort-alpha 0`, crop to valid ROI.

    Returns:
        undistorted_image, updated_cam_params
    """
    k_raw, dist = _get_radtan_distortion(cam_params)
    # keep legacy flags, but always use alpha=0 ROI crop for stable downstream geometry
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    undist_bgr, k_new, _ = undistort_image(image_bgr, k_raw, dist)
    undistorted = cv2.cvtColor(undist_bgr, cv2.COLOR_BGR2RGB)

    updated = dict(cam_params)
    updated["fx"] = float(k_new[0, 0])
    updated["fy"] = float(k_new[1, 1])
    updated["cx"] = float(k_new[0, 2])
    updated["cy"] = float(k_new[1, 2])
    updated["camera_model"] = "PINHOLE"
    return undistorted, updated


def _iter_image_paths(input_dir: str, pattern: Optional[str]) -> List[str]:
    if pattern is not None:
        return sorted(glob.glob(os.path.join(input_dir, pattern)))
    image_paths: List[str] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    return sorted(image_paths)


def run_custom_folder(model, model_name: str, device, config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.intrinsics is None:
        raise ValueError("--intrinsics is required")
    if args.input_dir is None:
        raise ValueError("--input-dir is required")

    cano_sz = config["cano_sz"]  # ERP size model was trained on
    cam_params_base = _load_intrinsics_json(args.intrinsics)

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"--input-dir not found: {args.input_dir}")

    depth_dir = os.path.join(args.out_dir, "depth_uint16")
    npy_dir = os.path.join(args.out_dir, "depth_npy")
    vis_dir = os.path.join(args.out_dir, "vis")
    overlay_dir = os.path.join(args.out_dir, "vis_overlay")
    _ensure_dir(depth_dir)
    if args.save_npy:
        _ensure_dir(npy_dir)
    if args.vis:
        _ensure_dir(vis_dir)
        _ensure_dir(overlay_dir)

    image_paths = _iter_image_paths(args.input_dir, args.glob)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir} (glob={args.glob!r})")

    fwd_sz = tuple(args.fwd_sz)
    if len(fwd_sz) != 2:
        raise ValueError("--fwd-sz must be two ints: H W")

    if args.output_downscale <= 0:
        raise ValueError("--output-downscale must be > 0")

    print(f"Found {len(image_paths)} images in {args.input_dir}")
    for idx, image_path in enumerate(image_paths):
        image_raw = np.asarray(Image.open(image_path))
        if image_raw.ndim == 2:
            image_raw = np.stack([image_raw, image_raw, image_raw], axis=-1)
        if image_raw.shape[2] == 4:
            image_raw = image_raw[:, :, :3]

        cam_params = cam_params_base
        image = image_raw

        k_raw, dist = _get_radtan_distortion(cam_params_base)
        has_dist = _has_nonzero_distortion(dist)
        if has_dist:
            if not args.undistort:
                print(f"[WARN] Detected non-zero pinhole distortion in intrinsics; undistorting for inference: {os.path.basename(image_path)}")
            image, cam_params = _undistort_pinhole(image_raw, cam_params_base, args)

        org_img_h, org_img_w = image.shape[:2]

        # Dummy "gt depth" only to reuse the existing ERP conversion pipeline.
        depth_dummy = np.ones((org_img_h, org_img_w, 1), dtype=np.float32)
        mask_valid_depth = np.ones_like(depth_dummy, dtype=np.float32)

        # Derive crop sizes from the (optimized) intrinsics after undistortion/cropping.
        crop_wfov = float(args.crop_wfov) if args.crop_wfov is not None else None
        crop_vfov = float(args.crop_vfov) if args.crop_vfov is not None else None
        if crop_wfov is None:
            crop_wfov = _infer_fov_deg_from_focal(float(cam_params["fx"]), int(org_img_w)) + float(args.crop_wfov_margin)
        if crop_vfov is None:
            crop_vfov = _infer_fov_deg_from_focal(float(cam_params["fy"]), int(org_img_h)) + float(args.crop_vfov_margin)

        crop_wfov = float(min(179.0, max(10.0, crop_wfov)))
        crop_vfov = float(min(179.0, max(10.0, crop_vfov)))

        phi = np.array(0).astype(np.float32)
        roll = np.array(0).astype(np.float32)
        theta = 0

        image_f = image.astype(np.float32) / 255.0

        crop_width = int(round(cano_sz[0] * crop_wfov / 180.0))
        crop_height = int(round(cano_sz[0] * crop_vfov / 180.0))
        crop_width = max(16, crop_width)
        crop_height = max(16, crop_height)

        # Undistorted pinhole -> ERP patch (model input)
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

        # Resize ERP patch to model input size
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

        with torch.no_grad():
            if model_name == "IDiscERP":
                preds, _, _ = model(image_t.unsqueeze(0).to(device), lat_range.unsqueeze(0).to(device), long_range.unsqueeze(0).to(device))
            else:
                preds, _, _ = model(image_t.unsqueeze(0).to(device))
        preds *= pred_scale_factor

        # Inverse mapping ERP patch -> pinhole image for visualization/output
        erp_h = cano_sz[0] * pred_scale_factor

        out_h = int(org_img_h / args.output_downscale)
        out_w = int(org_img_w / args.output_downscale)
        _, depth_out, _, active_mask = erp_patch_to_cam_fast(
            image_t,
            preds[0].detach().cpu(),
            attn_mask_t,
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

        base = os.path.splitext(os.path.basename(image_path))[0]

        depth_m = depth_out.squeeze().numpy().astype(np.float32)
        active_mask_np = active_mask.squeeze().numpy().astype(np.float32)

        valid = np.isfinite(depth_m) & (depth_m > 0) & (active_mask_np > 0)
        if np.any(valid):
            max_depth_m = float(np.max(depth_m[valid]))
            max_scaled = max_depth_m * float(args.depth_scale)
            if max_scaled > 65535:
                safe_scale = int(65535.0 / max(1e-6, max_depth_m))
                print(
                    f"[WARN] depth uint16 saturation risk: max_depth~{max_depth_m:.2f}m, "
                    f"depth_scale={args.depth_scale} -> {max_scaled:.1f} (>65535). "
                    f"Consider --depth-scale <= {safe_scale} (or enable --save-npy)."
                )

        depth_uint16 = np.clip(depth_m * float(args.depth_scale), 0, 65535).astype(np.uint16)
        depth_path = os.path.join(depth_dir, f"{base}.png")
        cv2.imwrite(depth_path, depth_uint16)

        if args.save_npy:
            np.save(os.path.join(npy_dir, f"{base}.npy"), depth_m)

        if args.vis:
            rgb_vis = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            rgb_vis_t = TF.normalize(TF.to_tensor(rgb_vis), **normalization_stats)
            depth_vis_t = torch.from_numpy(depth_m).unsqueeze(0)
            active_mask_t_vis = torch.from_numpy(active_mask_np).unsqueeze(0)
            save_val_imgs_metric_values(
                depth_vis_t,
                rgb_vis_t,
                f"{base}_vis.jpg",
                vis_dir,
                active_mask=active_mask_t_vis,
                depth_max=args.vis_depth_max,
            )
            # Overlay depth on the (undistorted) RGB image to visually inspect alignment.
            rgb_vis_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
            overlay_path = os.path.join(overlay_dir, f"{base}_overlay.jpg")
            _save_depth_overlay_bgr(
                rgb_vis_bgr,
                depth_m,
                active_mask_np,
                overlay_path,
                depth_max=args.vis_depth_max,
                overlay_alpha=float(args.overlay_alpha),
            )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] saved {depth_path}")


def _load_model(config: Dict[str, Any], model_file: str):
    """
    Lazily import model modules so `--help` works without optional compiled ops.
    """
    from dac.models.cnn_depth import CNNDepth  # noqa: F401
    from dac.models.idisc import IDisc  # noqa: F401
    from dac.models.idisc_erp import IDiscERP  # noqa: F401
    from dac.models.idisc_equi import IDiscEqui  # noqa: F401

    model = eval(config["model_name"]).build(config)
    model.load_pretrained(model_file)
    return model


def main(config: Dict[str, Any], args: argparse.Namespace) -> None:
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = _load_model(config, args.model_file)
    model = model.to(device)
    model.eval()
    run_custom_folder(model, config["model_name"], device, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAC custom outdoor demo", conflict_handler="resolve")
    parser.add_argument("--config-file", type=str, required=True, help="Model config JSON.")
    parser.add_argument("--model-file", type=str, required=True, help="Model weights.")
    parser.add_argument("--out-dir", type=str, default="demo/output_custom")

    parser.add_argument("--input-dir", type=str, required=True, help="Input image folder.")
    parser.add_argument("--intrinsics", type=str, required=True, help="Camera intrinsics JSON for pinhole images.")
    parser.add_argument("--glob", type=str, default=None, help="Optional glob pattern inside --input-dir (e.g. '*.jpg').")

    parser.add_argument("--fwd-sz", type=int, nargs=2, default=[576, 1024], metavar=("H", "W"), help="Model input patch size (H W).")
    parser.add_argument("--crop-wfov", type=float, default=None, help="Horizontal crop FoV in degrees. If omitted, inferred from optimized fx and undistorted width.")
    parser.add_argument("--crop-vfov", type=float, default=None, help="Vertical crop FoV in degrees. If omitted, inferred from optimized fy and undistorted height.")
    parser.add_argument("--crop-wfov-margin", type=float, default=10.0, help="Extra degrees added to inferred horizontal FoV.")
    parser.add_argument("--crop-vfov-margin", type=float, default=2.0, help="Extra degrees added to inferred vertical FoV (helps avoid top/bottom arc cut).")

    parser.add_argument("--output-downscale", type=float, default=1.0, help="Downscale output depth resolution (e.g. 2 -> half-res).")
    parser.add_argument("--depth-scale", type=int, default=1000, help="Scale factor for uint16 depth PNG (depth[m] * depth_scale).")
    parser.add_argument("--save-npy", action="store_true", help="Also save float32 depth in meters as .npy.")
    parser.add_argument("--vis", action="store_true", help="Save RGB|depth visualization images.")
    parser.add_argument("--vis-depth-max", type=float, default=None, help="Visualization max depth (meters).")
    parser.add_argument("--overlay-alpha", type=float, default=0.55, help="Alpha for depth overlay visualization.")

    parser.add_argument("--undistort", action="store_true", help="Enable undistortion using k1,k2,p1,p2,k3 from --intrinsics (OpenCV radtan).")
    parser.add_argument("--undistort-optimal", action="store_true", help="Use getOptimalNewCameraMatrix for undistortion.")
    parser.add_argument("--undistort-alpha", type=float, default=0.0, help="Alpha for getOptimalNewCameraMatrix (0=crop, 1=keep all pixels).")

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)
    main(config, args)
