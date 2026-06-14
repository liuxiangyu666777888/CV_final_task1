#!/usr/bin/env python3
"""
render_video.py — Render a multi-view roaming video from merged gaussian splats.

Uses gsplat (from nerfstudio) for CUDA-accelerated rendering, then ffmpeg to
compose frames into an MP4 video.

Usage:
    python scripts/fusion/render_video.py --config configs/fusion.yaml

Requirements:
    pip install gsplat imageio imageio-ffmpeg
"""

import argparse
import os
import re
import shutil
import sys
import yaml
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.io_utils import load_gaussian_ply
from utils.camera_utils import (
    estimate_orientation_from_colmap,
    generate_circle_path,
    generate_spiral_path,
    get_projection_matrix,
)


def parse_source_path_from_cfg_args(cfg_args_path: str) -> str | None:
    """Extract `source_path` from a saved argparse Namespace string."""
    if not os.path.exists(cfg_args_path):
        return None
    with open(cfg_args_path, "r") as f:
        text = f.read()
    match = re.search(r"source_path=['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else None


def infer_orientation_scene_path(config: dict, camera_path_cfg: dict) -> tuple[str | None, str | None]:
    """
    Resolve a COLMAP scene path for auto-orientation.

    Preference order:
      1. `camera_path.orientation_scene`
      2. background model's saved `cfg_args -> source_path`
    """
    explicit_scene = camera_path_cfg.get("orientation_scene")
    if explicit_scene:
        return explicit_scene, "camera_path.orientation_scene"

    bg_ply_path = config.get("background", {}).get("ply_path")
    if not bg_ply_path:
        return None, None

    search_dir = os.path.abspath(os.path.dirname(bg_ply_path))
    while True:
        cfg_args_path = os.path.join(search_dir, "cfg_args")
        source_path = parse_source_path_from_cfg_args(cfg_args_path)
        if source_path:
            return source_path, cfg_args_path
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir

    return None, None


def configure_gsplat_build_env() -> None:
    """Pin gsplat's CUDA extension build to a compiler compatible with CUDA 12.4."""
    gcc11 = shutil.which("gcc-11")
    gxx11 = shutil.which("g++-11")

    if gcc11 and "CC" not in os.environ:
        os.environ["CC"] = gcc11
    if gxx11:
        os.environ.setdefault("CXX", gxx11)
        os.environ.setdefault("CUDAHOSTCXX", gxx11)

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            import torch

            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        except Exception:
            pass


def render_frame_gsplat(
    gs_data: dict,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    coord_flip_mode: str = "yz",
) -> np.ndarray:
    """
    Render a single frame using gsplat.

    Uses the rasterizer from gsplat to project 3D gaussians → 2D image.

    Args:
        gs_data: dict from load_gaussian_ply()
        view_matrix:  (4, 4) world-to-camera matrix
        proj_matrix:  (4, 4) projection matrix
        width, height: output resolution

    Returns:
        (H, W, 3) uint8 rendered image
    """
    try:
        configure_gsplat_build_env()
        from gsplat import rasterization
    except ImportError:
        raise ImportError(
            "gsplat is required. Install: pip install gsplat"
        )

    # Prepare tensors
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    means = torch.from_numpy(gs_data["xyz"]).float().to(device)
    quats = torch.from_numpy(gs_data["rotation"]).float().to(device)  # (N, 4) wxyz
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.from_numpy(gs_data["scaling"]).float().to(device).exp()
    opacities = torch.from_numpy(gs_data["opacity"]).float().squeeze(-1).to(device).sigmoid()
    colors = torch.from_numpy(gs_data["features_dc"]).float().to(device).squeeze(-1)
    colors = torch.clamp(colors * 0.28209479177387814 + 0.5, 0.0, 1.0)

    viewmat = torch.from_numpy(view_matrix).float().to(device)  # (4, 4)
    # camera_utils uses an OpenGL-style camera where points in front have negative z.
    # gsplat expects OpenCV-style camera coordinates with positive z forward.
    flip_mats = {
        "none": torch.eye(4, device=device),
        "z": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            device=device,
        ),
        "y": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            device=device,
        ),
        "yz": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, -1.0, 0.0, 0.0],
             [0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            device=device,
        ),
    }
    coord_flip = flip_mats.get(coord_flip_mode, flip_mats["yz"])
    viewmat = coord_flip @ viewmat
    projmat = torch.from_numpy(proj_matrix).float().to(device)

    K = torch.eye(3, device=device)
    K[0, 0] = width / 2.0
    K[1, 1] = height / 2.0
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0

    # gsplat rasterization
    rendered, _, _ = rasterization(
        means=means.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        colors=colors.unsqueeze(0),
        viewmats=viewmat.unsqueeze(0).unsqueeze(0),
        Ks=K.unsqueeze(0).unsqueeze(0),
        width=width,
        height=height,
    )

    img = rendered.squeeze(0).squeeze(0).detach().cpu().numpy()  # (H, W, 3)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    return img


def render_fallback(
    gs_data: dict,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Fallback software renderer using simple point projection.
    Used when gsplat is not available — produces approximate result.
    """
    xyz = gs_data["xyz"].astype(np.float32, copy=False)
    colors = gs_data["features_dc"].reshape(-1, 3).astype(np.float32, copy=False)
    colors = np.clip(colors * 0.28209479177387814 + 0.5, 0, 1)  # SH DC -> RGB
    opacity_raw = gs_data["opacity"].reshape(-1).astype(np.float32, copy=False)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(opacity_raw, -30.0, 30.0)))

    # Keep fallback tractable on large scenes.
    max_points = 400_000
    if xyz.shape[0] > max_points:
        step = int(np.ceil(xyz.shape[0] / max_points))
        idx = np.arange(0, xyz.shape[0], step)
        xyz = xyz[idx]
        colors = colors[idx]
        opacity = opacity[idx]

    # Transform to camera space
    xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
    cam_xyz = (view_matrix @ xyz_h.T).T[:, :3]

    # Simple pinhole projection
    fx = width / 2.0
    fy = height / 2.0
    cx = width / 2.0
    cy = height / 2.0

    z = cam_xyz[:, 2]
    # View matrices from camera_utils follow the OpenGL convention where
    # points in front of the camera have negative z in camera space.
    valid = (z < -0.01) & (opacity > 0.02)
    depth = -z[valid]
    u = (cam_xyz[valid, 0] * fx / depth + cx).astype(int)
    v = (cam_xyz[valid, 1] * fy / depth + cy).astype(int)

    img = np.ones((height, width, 3), dtype=np.float32) * 0.08
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[in_frame], v[in_frame]
    colors_valid = colors[np.where(valid)[0]][in_frame]
    opacity_valid = opacity[np.where(valid)[0]][in_frame]
    z_valid = depth[in_frame]
    if len(u) == 0:
        return (img * 255).astype(np.uint8)

    # Z-buffer by pixel, vectorized: keep the closest projected point per pixel.
    pix = v * width + u
    order = np.lexsort((z_valid, pix))
    pix_sorted = pix[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = pix_sorted[1:] != pix_sorted[:-1]
    chosen = order[first]

    color_map = np.zeros((height, width, 3), dtype=np.float32)
    alpha_map = np.zeros((height, width), dtype=np.float32)
    color_map[v[chosen], u[chosen]] = colors_valid[chosen]
    alpha_map[v[chosen], u[chosen]] = np.clip(opacity_valid[chosen], 0.0, 1.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_map = cv2.dilate(alpha_map, kernel, iterations=1)
    blurred_alpha = cv2.GaussianBlur(alpha_map, (0, 0), sigmaX=1.2, sigmaY=1.2)

    blurred_color = np.zeros_like(color_map)
    for c in range(3):
        weighted = cv2.GaussianBlur(
            color_map[:, :, c] * alpha_map,
            (0, 0),
            sigmaX=1.2,
            sigmaY=1.2,
        )
        blurred_color[:, :, c] = weighted

    norm = np.clip(blurred_alpha[..., None], 1e-6, None)
    splat_rgb = np.clip(blurred_color / norm, 0.0, 1.0)
    splat_alpha = np.clip(blurred_alpha[..., None] * 1.6, 0.0, 1.0)
    img = img * (1.0 - splat_alpha) + splat_rgb * splat_alpha

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_frame(
    gs_data: dict,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: str,
    coord_flip_mode: str,
) -> tuple[np.ndarray, str]:
    if backend == "gsplat":
        try:
            return (
                render_frame_gsplat(
                    gs_data,
                    view_matrix,
                    proj_matrix,
                    width,
                    height,
                    coord_flip_mode=coord_flip_mode,
                ),
                "gsplat",
            )
        except Exception as exc:
            print(f"[render] WARNING: gsplat backend failed, switching to fallback: {exc}")
            return render_fallback(gs_data, view_matrix, proj_matrix, width, height), "fallback"
    return render_fallback(gs_data, view_matrix, proj_matrix, width, height), "fallback"


def apply_output_crop(img: np.ndarray, crop_cfg: dict) -> np.ndarray:
    """
    Crop away unstable frame borders, then resize back to the target canvas.

    Fractions are relative to the rendered frame size.
    """
    if not crop_cfg:
        return img

    h, w = img.shape[:2]
    top = int(round(h * float(crop_cfg.get("top_frac", 0.0))))
    bottom = int(round(h * float(crop_cfg.get("bottom_frac", 0.0))))
    left = int(round(w * float(crop_cfg.get("left_frac", 0.0))))
    right = int(round(w * float(crop_cfg.get("right_frac", 0.0))))

    y0 = max(0, min(h - 1, top))
    y1 = max(y0 + 1, min(h, h - bottom))
    x0 = max(0, min(w - 1, left))
    x1 = max(x0 + 1, min(w, w - right))

    cropped = img[y0:y1, x0:x1]
    if cropped.shape[0] == h and cropped.shape[1] == w:
        return img
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def repair_top_dark_border(img: np.ndarray, repair_cfg: dict) -> np.ndarray:
    """
    Inpaint dark holes connected to the top frame edge.

    SfM/2DGS backgrounds often have no reliable splats for sky or image-border
    regions. Restricting the mask to top-connected dark components avoids
    modifying legitimate dark content such as windows.
    """
    if not repair_cfg or not repair_cfg.get("enabled", False):
        return img

    h, w = img.shape[:2]
    max_rows = int(round(h * float(repair_cfg.get("top_max_frac", 0.22))))
    max_rows = max(1, min(h, max_rows))
    threshold = float(repair_cfg.get("dark_threshold", 24.0))

    gray = img[:max_rows].mean(axis=2)
    dark = (gray <= threshold).astype(np.uint8)
    if int(dark[0].sum()) == 0:
        return img

    num_labels, labels = cv2.connectedComponents(dark, connectivity=8)
    top_labels = np.unique(labels[0][dark[0] > 0])
    top_labels = top_labels[top_labels != 0]
    if top_labels.size == 0:
        return img

    top_mask = np.isin(labels, top_labels).astype(np.uint8)
    min_area = int(repair_cfg.get("min_area", 128))
    if int(top_mask.sum()) < min_area:
        return img

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[:max_rows] = top_mask * 255

    dilate_iters = int(repair_cfg.get("dilate_iters", 2))
    if dilate_iters > 0:
        kernel_size = int(repair_cfg.get("kernel_size", 5))
        kernel_size = max(3, kernel_size | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        full_mask = cv2.dilate(full_mask, kernel, iterations=dilate_iters)

    radius = float(repair_cfg.get("inpaint_radius", 5.0))
    repaired = cv2.inpaint(img, full_mask, radius, cv2.INPAINT_TELEA)
    return repaired


def composite_foreground_overlay(
    base_img: np.ndarray,
    overlay_img: np.ndarray,
    overlay_cfg: dict,
) -> np.ndarray:
    """Composite a separately rendered foreground layer over the base frame."""
    if not overlay_cfg or not overlay_cfg.get("enabled", False):
        return base_img

    threshold = float(overlay_cfg.get("mask_threshold", 8.0))
    alpha = (overlay_img.mean(axis=2) > threshold).astype(np.float32)
    if not np.any(alpha):
        return base_img

    dilate_iters = int(overlay_cfg.get("dilate_iters", 1))
    if dilate_iters > 0:
        kernel_size = int(overlay_cfg.get("kernel_size", 3))
        kernel_size = max(3, kernel_size | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        alpha = cv2.dilate(alpha, kernel, iterations=dilate_iters)

    blur_sigma = float(overlay_cfg.get("blur_sigma", 0.8))
    if blur_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    opacity = float(overlay_cfg.get("opacity", 1.0))
    alpha = np.clip(alpha[..., None] * opacity, 0.0, 1.0)
    out = base_img.astype(np.float32) * (1.0 - alpha) + overlay_img.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Render multi-view roaming video from merged GS PLY"
    )
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--ply", default="outputs/merged.ply",
                        help="Merged PLY file path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--resolution", default=None, help="WIDTH,HEIGHT")
    parser.add_argument("--backend", default="gsplat",
                        choices=["gsplat", "fallback"],
                        help="Rendering backend")
    parser.add_argument("--coord-flip", default=None,
                        choices=["none", "z", "y", "yz"],
                        help="Override gsplat camera coordinate flip mode")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ply_path = args.ply
    cp = config["camera_path"]
    out = config["output"]

    num_frames = args.frames or cp.get("frames") or out.get("frames", 300)
    resolution_str = args.resolution
    if resolution_str:
        width, height = map(int, resolution_str.split(","))
    else:
        width, height = out.get("resolution", [1920, 1080])
    fps = out.get("fps", 30)
    frames_dir = out.get("frames_dir", "outputs/frames")
    video_path = args.output or out.get("video_path", "outputs/roaming.mp4")
    crop_cfg = out.get("crop", {})
    border_repair_cfg = out.get("border_repair", {})
    foreground_overlay_cfg = out.get("foreground_overlay", {})

    os.makedirs(frames_dir, exist_ok=True)
    for name in os.listdir(frames_dir):
        if name.startswith("frame_") and name.endswith(".png"):
            os.remove(os.path.join(frames_dir, name))

    # Load gaussians
    print(f"[render] Loading merged PLY: {ply_path}")
    gs_data = load_gaussian_ply(ply_path)
    print(f"  Total gaussians: {gs_data['xyz'].shape[0]}")
    foreground_data = None
    foreground_ply = out.get("foreground_ply")
    if foreground_overlay_cfg.get("enabled", False) and foreground_ply:
        if os.path.exists(foreground_ply):
            print(f"[render] Loading foreground overlay PLY: {foreground_ply}")
            foreground_data = load_gaussian_ply(foreground_ply)
            print(f"  Foreground gaussians: {foreground_data['xyz'].shape[0]}")
        else:
            print(f"[render] WARNING: foreground overlay PLY not found: {foreground_ply}")

    # Generate camera path
    path_type = cp.get("type", "spiral")
    print(f"[render] Generating camera path: {path_type}")
    look_at = np.array(cp.get("look_at", [0.0, 0.5, 0.0]), dtype=np.float32)
    up = np.array(cp.get("up", [0.0, 1.0, 0.0]), dtype=np.float32)
    start_dir = None

    should_auto_orient = bool(cp.get("auto_orient", False) or cp.get("orientation_scene"))
    if should_auto_orient:
        orientation_scene, orientation_source = infer_orientation_scene_path(config, cp)
        if orientation_scene:
            orientation = estimate_orientation_from_colmap(orientation_scene)
            print(f"[render] Auto orientation from: {orientation_scene}")
            print(f"  resolved via: {orientation_source}")
            print(f"  inferred up: {np.round(orientation['up'], 4).tolist()}")
            print(f"  focus point: {np.round(orientation['focus_point'], 4).tolist()}")
            print(
                "  training camera radius mean/min/max: "
                f"{float(orientation['orbit_radius_mean']):.3f} / "
                f"{float(orientation['orbit_radius_min']):.3f} / "
                f"{float(orientation['orbit_radius_max']):.3f}"
            )
            print(
                "  training camera height mean/std: "
                f"{float(orientation['height_mean']):.3f} / "
                f"{float(orientation['height_std']):.3f}"
            )
            if "up" not in cp:
                up = orientation["up"]
            if "look_at" not in cp:
                look_at = orientation["focus_point"]
            start_dir = orientation["camera_centers"][0] - look_at
        else:
            print("[render] WARNING: auto_orient requested but no COLMAP scene was resolved.")

    print(f"  using look_at: {np.round(look_at, 4).tolist()}")
    print(f"  using up: {np.round(up, 4).tolist()}")
    if path_type == "circle":
        view_mats = generate_circle_path(
            num_frames=num_frames,
            radius=cp.get("radius", 3.0),
            height=cp.get("height", 1.0),
            angle_offset_deg=cp.get("angle_offset_deg", 0.0),
            sweep_deg=cp.get("sweep_deg", 360.0),
            look_at=tuple(look_at.tolist()),
            up=tuple(up.tolist()),
            start_dir=None if start_dir is None else tuple(start_dir.tolist()),
        )
    else:
        view_mats = generate_spiral_path(
            num_frames=num_frames,
            radius=cp.get("radius", 3.0),
            height_start=cp.get("height_start", 0.5),
            height_end=cp.get("height_end", 2.0),
            turns=cp.get("turns", 2),
            angle_offset_deg=cp.get("angle_offset_deg", 0.0),
            look_at=tuple(look_at.tolist()),
            up=tuple(up.tolist()),
            start_dir=None if start_dir is None else tuple(start_dir.tolist()),
        )

    proj_mat = get_projection_matrix(aspect=width / height)
    coord_flip_mode = args.coord_flip or out.get("gsplat_coord_flip", "yz")

    # Render frames
    print(f"[render] Rendering {num_frames} frames @ {width}×{height}...")
    active_backend = args.backend
    for i, view_mat in enumerate(view_mats):
        img, active_backend = render_frame(
            gs_data, view_mat, proj_mat, width, height, active_backend, coord_flip_mode
        )
        img = repair_top_dark_border(img, border_repair_cfg)
        img = apply_output_crop(img, crop_cfg)
        if foreground_data is not None:
            overlay_img, active_backend = render_frame(
                foreground_data,
                view_mat,
                proj_mat,
                width,
                height,
                active_backend,
                coord_flip_mode,
            )
            overlay_img = repair_top_dark_border(overlay_img, border_repair_cfg)
            overlay_img = apply_output_crop(overlay_img, crop_cfg)
            img = composite_foreground_overlay(img, overlay_img, foreground_overlay_cfg)
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        Image.fromarray(img).save(frame_path)

        if (i + 1) % 50 == 0 or i == num_frames - 1:
            print(f"  [{i+1}/{num_frames}] frames rendered")

    # Compose video with ffmpeg
    print(f"[render] Composing video with ffmpeg...")
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%05d.png "
        f"-c:v libx264 -pix_fmt yuv420p -crf 18 {video_path}"
    )
    os.system(ffmpeg_cmd)

    print(f"[render] Done! Video saved to: {video_path}")
    print(f"  Duration: {num_frames / fps:.1f}s, {fps} fps")


if __name__ == "__main__":
    main()
