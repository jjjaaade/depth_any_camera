#!/usr/bin/env python
"""
Batch depth inference for custom outdoor data, based on demo_dac_custom_outdoor.py.

Features:
  - Read a txt list of relative directories under --root-prefix
  - Resolve per-folder calibration from --calib-dir (plate-based), load camera intrinsics/distortion from yml
  - Undistort image with OpenCV radtan (alpha=0 ROI crop) and run DAC inference
  - Save uint16 depth PNG to a per-image-folder `depth/` directory
  - Randomly sample a subset of results for visualization
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.cuda as tcuda
import torchvision.transforms.functional as TF
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dac.dataloders.dataset import resize_for_input
from dac.utils.erp_geometry import cam_to_erp_patch_fast, erp_patch_to_cam_fast
from dac.utils.visualization import save_val_imgs_metric_values

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

CAMERA_INFOS = {
    "05": {"yaml": "camera05_2_left_front_m1"},
    "06": {"yaml": "camera06_2_left_rear_m1"},
    "07": {"yaml": "camera07_2_right_rear_m1"},
    "08": {"yaml": "camera08_2_right_front_m1"},
    "09": {"yaml": "camera09_2_rear_m1"},
}

CAM08_YAML_NAME = "camera08_2_right_front_m1"


def read_relative_dirs(txt_path: str) -> list[str]:
    rels: list[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rels.append(s)
    return rels


def split_plate_list(plate_list: list[str]) -> tuple[list[str], list[str]]:
    char_plate_list: list[str] = []
    num_plate_list: list[str] = []
    for p in plate_list:
        if not p.isdigit():
            char_plate_list.append(p)
        else:
            num_plate_list.append(p)
    return sorted(char_plate_list), sorted(num_plate_list)


def normalize_plate_name(plate: Optional[str]) -> Optional[str]:
    if plate is None:
        return None
    if plate.startswith("ym"):
        return plate.replace("ym", "AD", 1)
    if plate.startswith("tongli"):
        return plate.replace("tongli", "JNW", 1)
    return plate


def build_plate_lists(calib_dir: str) -> tuple[list[str], list[str]]:
    plates_list = [
        name for name in os.listdir(calib_dir) if os.path.isdir(os.path.join(calib_dir, name))
    ]
    return split_plate_list(plates_list)


def match_plate_from_path(path_str: str, char_plates_list: list[str], num_plates_list: list[str]) -> Optional[str]:
    parts = [p for p in re.split(r"[\\/]+", path_str) if p]
    part_set_lower = {p.lower() for p in parts}

    for plate in char_plates_list:
        if plate.lower() in part_set_lower:
            return plate
    for plate in num_plates_list:
        if plate in parts:
            return plate

    for plate in sorted(char_plates_list, key=len, reverse=True):
        if re.search(rf"(^|[\\/]){re.escape(plate)}([\\/]|$)", path_str, flags=re.IGNORECASE):
            return plate
    for plate in sorted(num_plates_list, key=len, reverse=True):
        if re.search(rf"(^|[\\/]){re.escape(plate)}([\\/]|$)", path_str):
            return plate

    return None


def resolve_target_calib_dir(
    dir_path: Path,
    calib_dir: Path,
    char_plates_list: list[str],
    num_plates_list: list[str],
) -> Optional[Path]:
    """
    Try to resolve plate from a json file in dir_path, then map to calib_dir/plate.
    Fallback to path matching if json/plate unavailable.
    """
    json_path = dir_path / (dir_path.name + ".json")
    json_data: Optional[dict[str, Any]] = None

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception:
            json_data = None

    plate: Optional[str] = None
    if json_data and isinstance(json_data, dict):
        custom_infos = json_data.get("custom_infos", None)
        if isinstance(custom_infos, dict):
            plate = custom_infos.get("plate", None)

    plate = normalize_plate_name(plate)
    if plate not in char_plates_list and plate not in num_plates_list:
        plate = None

    if plate is None:
        plate = match_plate_from_path(str(dir_path), char_plates_list, num_plates_list)

    if plate is not None:
        target = calib_dir / plate
        if target.is_dir():
            return target
    return None


def read_calibration_files(target_calib_dir: Path) -> dict[str, cv2.FileStorage]:
    calibrations: dict[str, cv2.FileStorage] = {}
    if not target_calib_dir:
        return calibrations

    for root, _, files in os.walk(str(target_calib_dir)):
        for file_name in files:
            if not file_name.endswith(".yml"):
                continue
            basename = os.path.splitext(file_name)[0]
            calib_path = os.path.join(root, file_name)
            fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
            if fs.isOpened():
                calibrations[basename] = fs
    return calibrations


def release_calibrations(calibrations: dict[str, cv2.FileStorage]) -> None:
    for fs in calibrations.values():
        try:
            fs.release()
        except Exception:
            pass


def parse_cam_id_from_name(name: str) -> Optional[str]:
    patterns = [
        r"camera[_\-]?0?([5-9])",
        r"cam[_\-]?0?([5-9])",
    ]
    lower = name.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            return "0" + m.group(1) if len(m.group(1)) == 1 else m.group(1)
    return None


def list_candidate_images(
    input_dir: Path,
    recursive: bool,
    name_contains: Optional[str],
    exts: set[str],
) -> list[tuple[Path, str]]:
    files = input_dir.rglob("*") if recursive else input_dir.iterdir()
    results: list[tuple[Path, str]] = []
    for p in files:
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if name_contains and name_contains not in p.name:
            continue
        if {"sky_mask", "undistort", "depth"}.intersection(set(p.parts)):
            continue
        cam_id = parse_cam_id_from_name(p.name)
        if cam_id is None:
            continue
        results.append((p, cam_id))
    return sorted(results, key=lambda x: str(x[0]))


def get_camera_calib_for_cam(
    calibrations: dict[str, cv2.FileStorage],
    cam_id: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    cam08 must exist.
    other cams fallback to cam08 if missing.
    """
    if CAM08_YAML_NAME not in calibrations:
        raise ValueError(f"Required calibration missing: {CAM08_YAML_NAME}")

    yaml_name = CAMERA_INFOS.get(cam_id, {}).get("yaml", CAM08_YAML_NAME)
    if yaml_name not in calibrations:
        yaml_name = CAM08_YAML_NAME

    fs = calibrations[yaml_name]
    camera_inner_parameter = fs.getNode("CameraMat").mat()
    camera_dist = fs.getNode("DistCoeff").mat()
    if camera_inner_parameter is None or camera_dist is None:
        raise ValueError(f"Invalid calibration data in {yaml_name}")
    return camera_inner_parameter, camera_dist, yaml_name


def undistort_image(
    image_bgr: np.ndarray,
    camera_mat: np.ndarray,
    dist_coeff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Undistort with alpha=0 ROI crop to remove black borders, and adjust intrinsics.

    Returns:
        undistorted_bgr, optimal_intrinsic (cropped)
    """
    h, w = image_bgr.shape[:2]
    optimal_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
        np.array(camera_mat, dtype=np.float32),
        np.array(dist_coeff, dtype=np.float32),
        (int(w), int(h)),
        alpha=0.0,
        newImgSize=(int(w), int(h)),
    )
    undist_full = cv2.undistort(
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
        undist = undist_full[y0:y1, x0:x1]
        optimal_intrinsic = optimal_intrinsic.copy()
        optimal_intrinsic[0, 2] -= float(x0)
        optimal_intrinsic[1, 2] -= float(y0)
    else:
        undist = undist_full

    return undist, optimal_intrinsic.astype(np.float32)


def _infer_fov_deg_from_focal(focal_px: float, size_px: int) -> float:
    return 2.0 * math.degrees(math.atan(float(size_px) / (2.0 * float(focal_px))))


def _save_depth_overlay_bgr(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    active_mask: np.ndarray,
    out_path: Path,
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
        cv2.imwrite(str(out_path), rgb_bgr)
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
    cv2.imwrite(str(out_path), blended)


def _load_model(config: dict[str, Any], model_file: str):
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


def _should_save_vis_by_ratio(rng: random.Random, vis_ratio: float) -> bool:
    if vis_ratio <= 0:
        return False
    if vis_ratio >= 1:
        return True
    return rng.random() < vis_ratio


def _run_one_image(
    *,
    model,
    model_name: str,
    device,
    config: dict[str, Any],
    image_path: Path,
    cam_id: str,
    camera_mat: np.ndarray,
    dist_coeff: np.ndarray,
    args: argparse.Namespace,
    save_vis: bool,
) -> dict[str, Any]:
    depth_dir = image_path.parent / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    depth_png_path = depth_dir / f"{image_path.stem}.png"
    depth_npy_path = depth_dir / f"{image_path.stem}.npy"

    if depth_png_path.exists() and not args.overwrite:
        return {"status": "skip", "image_path": str(image_path), "depth_path": str(depth_png_path)}

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"status": "fail", "image_path": str(image_path), "reason": "imread failed"}

    undist_bgr, k_new = undistort_image(image_bgr, camera_mat, dist_coeff)
    undist_rgb = cv2.cvtColor(undist_bgr, cv2.COLOR_BGR2RGB)

    org_img_h, org_img_w = undist_rgb.shape[:2]
    cam_params: dict[str, Any] = {
        "dataset": "custom",
        "camera_model": "PINHOLE",
        "fx": float(k_new[0, 0]),
        "fy": float(k_new[1, 1]),
        "cx": float(k_new[0, 2]),
        "cy": float(k_new[1, 2]),
    }

    cano_sz = config["cano_sz"]
    fwd_sz = tuple(args.fwd_sz)

    depth_dummy = np.ones((org_img_h, org_img_w, 1), dtype=np.float32)
    mask_valid_depth = np.ones_like(depth_dummy, dtype=np.float32)

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

    image_f = undist_rgb.astype(np.float32) / 255.0

    crop_width = int(round(cano_sz[0] * crop_wfov / 180.0))
    crop_height = int(round(cano_sz[0] * crop_vfov / 180.0))
    crop_width = max(16, crop_width)
    crop_height = max(16, crop_height)

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
            preds, _, _ = model(
                image_t.unsqueeze(0).to(device),
                lat_range.unsqueeze(0).to(device),
                long_range.unsqueeze(0).to(device),
            )
        else:
            preds, _, _ = model(image_t.unsqueeze(0).to(device))
    preds *= pred_scale_factor

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

    depth_m = depth_out.squeeze().numpy().astype(np.float32)
    active_mask_np = active_mask.squeeze().numpy().astype(np.float32)

    depth_uint16 = np.clip(depth_m * float(args.depth_scale), 0, 65535).astype(np.uint16)
    ok = cv2.imwrite(str(depth_png_path), depth_uint16)
    if not ok:
        return {"status": "fail", "image_path": str(image_path), "reason": "imwrite depth failed"}

    if args.save_npy:
        np.save(str(depth_npy_path), depth_m)

    if save_vis:
        vis_dir = depth_dir / "vis"
        overlay_dir = depth_dir / "vis_overlay"
        vis_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        rgb_vis = cv2.resize(undist_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        rgb_vis_t = TF.normalize(TF.to_tensor(rgb_vis), **normalization_stats)
        depth_vis_t = torch.from_numpy(depth_m).unsqueeze(0)
        active_mask_t_vis = torch.from_numpy(active_mask_np).unsqueeze(0)
        save_val_imgs_metric_values(
            depth_vis_t,
            rgb_vis_t,
            f"{image_path.stem}_vis.jpg",
            str(vis_dir),
            active_mask=active_mask_t_vis,
            depth_max=args.vis_depth_max,
        )

        overlay_path = overlay_dir / f"{image_path.stem}_overlay.jpg"
        rgb_vis_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
        _save_depth_overlay_bgr(
            rgb_vis_bgr,
            depth_m,
            active_mask_np,
            overlay_path,
            depth_max=args.vis_depth_max,
            overlay_alpha=float(args.overlay_alpha),
        )

    return {
        "status": "ok",
        "image_path": str(image_path),
        "cam_id": cam_id,
        "depth_path": str(depth_png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DAC custom outdoor batch inference", conflict_handler="resolve")
    parser.add_argument("--config-file", type=str, required=True, help="Model config JSON.")
    parser.add_argument("--model-file", type=str, required=True, help="Model weights.")

    parser.add_argument("--root-prefix", type=str, required=True, help="Root prefix for relative dirs in --txt-path.")
    parser.add_argument("--txt-path", type=str, required=True, help="Txt listing relative dirs, one per line.")
    parser.add_argument("--calib-dir", type=str, required=True, help="Calibration directory (contains plate subfolders).")

    parser.add_argument("--recursive", action="store_true", help="Recursively scan for images inside each listed dir.")
    parser.add_argument("--name-contains", type=str, default=None, help="Only process images whose basename contains this substring.")
    parser.add_argument(
        "--exts",
        type=str,
        default=",".join(SUPPORTED_IMAGE_EXTS),
        help="Comma-separated extensions to scan (e.g. .jpg,.png).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing depth outputs.")

    parser.add_argument("--fwd-sz", type=int, nargs=2, default=[576, 1024], metavar=("H", "W"), help="Model input patch size (H W).")
    parser.add_argument("--crop-wfov", type=float, default=None, help="Horizontal crop FoV in degrees. If omitted, inferred from optimized fx and undistorted width.")
    parser.add_argument("--crop-vfov", type=float, default=None, help="Vertical crop FoV in degrees. If omitted, inferred from optimized fy and undistorted height.")
    parser.add_argument("--crop-wfov-margin", type=float, default=10.0, help="Extra degrees added to inferred horizontal FoV.")
    parser.add_argument("--crop-vfov-margin", type=float, default=2.0, help="Extra degrees added to inferred vertical FoV.")

    parser.add_argument("--output-downscale", type=float, default=1.0, help="Downscale output depth resolution (e.g. 2 -> half-res).")
    parser.add_argument("--depth-scale", type=int, default=1000, help="Scale factor for uint16 depth PNG (depth[m] * depth_scale).")
    parser.add_argument("--save-npy", action="store_true", help="Also save float32 depth in meters as .npy alongside PNG.")

    parser.add_argument("--vis-ratio", type=float, default=0.0, help="Random sample ratio for saving visualizations (0 disables).")
    parser.add_argument("--vis-seed", type=int, default=0, help="RNG seed for visualization sampling.")
    parser.add_argument("--vis-depth-max", type=float, default=None, help="Visualization max depth (meters).")
    parser.add_argument("--overlay-alpha", type=float, default=0.55, help="Alpha for depth overlay visualization.")

    args = parser.parse_args()

    root_prefix = Path(args.root_prefix)
    calib_dir = Path(args.calib_dir)
    txt_path = Path(args.txt_path)

    if not root_prefix.is_dir():
        raise FileNotFoundError(f"--root-prefix not found: {root_prefix}")
    if not calib_dir.is_dir():
        raise FileNotFoundError(f"--calib-dir not found: {calib_dir}")
    if not txt_path.is_file():
        raise FileNotFoundError(f"--txt-path not found: {txt_path}")
    if args.output_downscale <= 0:
        raise ValueError("--output-downscale must be > 0")

    with open(args.config_file, "r") as f:
        config = json.load(f)

    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = _load_model(config, args.model_file).to(device)
    model.eval()

    char_plates_list, num_plates_list = build_plate_lists(str(calib_dir))
    exts = {x.strip().lower() for x in args.exts.split(",") if x.strip()}
    rng = random.Random(args.vis_seed)

    rel_dirs = read_relative_dirs(str(txt_path))
    abs_dirs = [root_prefix / rel for rel in rel_dirs]

    missing_dirs: list[str] = []
    missing_calib_dirs: list[str] = []
    tasks_by_calib_dir: dict[Path, list[tuple[Path, str]]] = defaultdict(list)
    image_count = 0

    print("Scanning input directories and resolving calibrations...")
    for abs_dir in tqdm(abs_dirs, desc="Scan dirs", unit="dir", dynamic_ncols=True):
        if not abs_dir.exists() or not abs_dir.is_dir():
            missing_dirs.append(str(abs_dir))
            continue

        target_calib_dir = resolve_target_calib_dir(
            abs_dir,
            calib_dir,
            char_plates_list,
            num_plates_list,
        )
        if target_calib_dir is None:
            missing_calib_dirs.append(str(abs_dir))
            continue

        image_items = list_candidate_images(
            input_dir=abs_dir,
            recursive=args.recursive,
            name_contains=args.name_contains,
            exts=exts,
        )
        for image_path, cam_id in image_items:
            tasks_by_calib_dir[target_calib_dir].append((image_path, cam_id))
        image_count += len(image_items)

    if missing_dirs:
        print(f"[WARN] Missing dirs: {len(missing_dirs)}")
    if missing_calib_dirs:
        print(f"[WARN] Missing calib dirs: {len(missing_calib_dirs)}")
    if image_count == 0:
        raise FileNotFoundError("No candidate images found.")

    ok_n = 0
    skip_n = 0
    fail_n = 0
    vis_n = 0

    print(f"Running inference on {image_count} images (grouped by {len(tasks_by_calib_dir)} calibration dirs)...")
    for target_calib_dir, items in tqdm(tasks_by_calib_dir.items(), desc="Calib groups", unit="group", dynamic_ncols=True):
        calibrations: dict[str, cv2.FileStorage] = {}
        try:
            calibrations = read_calibration_files(target_calib_dir)
            for image_path, cam_id in tqdm(items, desc=str(target_calib_dir.name), unit="img", dynamic_ncols=True, leave=False):
                camera_mat, dist_coeff, _used_yaml = get_camera_calib_for_cam(calibrations, cam_id)
                save_vis = _should_save_vis_by_ratio(rng, float(args.vis_ratio))
                if save_vis:
                    vis_n += 1
                result = _run_one_image(
                    model=model,
                    model_name=config["model_name"],
                    device=device,
                    config=config,
                    image_path=image_path,
                    cam_id=cam_id,
                    camera_mat=camera_mat,
                    dist_coeff=dist_coeff,
                    args=args,
                    save_vis=save_vis,
                )
                if result["status"] == "ok":
                    ok_n += 1
                elif result["status"] == "skip":
                    skip_n += 1
                else:
                    fail_n += 1
        finally:
            release_calibrations(calibrations)

    print(f"Done. ok={ok_n} skip={skip_n} fail={fail_n} vis_saved={vis_n}")


if __name__ == "__main__":
    main()
