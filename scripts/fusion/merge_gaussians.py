#!/usr/bin/env python3
"""
merge_gaussians.py — Load multiple Gaussian Splatting PLYs and merge them into one.

Applies per-object spatial transforms (translation, rotation, scale) then
concatenates all gaussians into a single output PLY file.

Usage:
    python scripts/fusion/merge_gaussians.py --config configs/fusion.yaml

Or with CLI overrides:
    python scripts/fusion/merge_gaussians.py \
        --bg outputs/background/point_cloud/iteration_30000/point_cloud.ply \
        --obj_a outputs/object_a.ply --pose_a 0.5,0.0,-0.5,0.3,0,45,0 \
        --obj_b outputs/object_b.ply --pose_b -0.8,0.3,0.2,0.5,0,0,0 \
        --obj_c outputs/object_c.ply --pose_c 1.0,0.0,0.8,0.4,0,-30,0 \
        --output outputs/merged.ply
"""

import argparse
import numpy as np
import os
import re
import sys
import yaml
from typing import Dict, Optional, Tuple
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.camera_utils import estimate_orientation_from_colmap
from utils.io_utils import load_gaussian_ply, save_gaussian_ply


def build_transform_matrix(
    translation: Tuple[float, float, float],
    scale: float,
    rotation_deg: Tuple[float, float, float],
) -> np.ndarray:
    """
    Build a 4×4 affine transform matrix from translation, scale, and Euler rotation.

    Args:
        translation:  (tx, ty, tz) in world units
        scale:        uniform scale factor
        rotation_deg: (rx, ry, rz) Euler angles in degrees

    Returns:
        (4, 4) transform matrix
    """
    rx, ry, rz = np.radians(rotation_deg)
    rot = R.from_euler("xyz", [rx, ry, rz]).as_matrix()

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot * scale
    T[:3, 3] = translation
    return T


def parse_source_path_from_cfg_args(cfg_args_path: str) -> Optional[str]:
    """Extract `source_path` from a saved argparse Namespace string."""
    if not os.path.exists(cfg_args_path):
        return None
    with open(cfg_args_path, "r") as f:
        text = f.read()
    match = re.search(r"source_path=['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else None


def infer_orientation_scene_path(config: dict) -> Optional[str]:
    """Resolve the background COLMAP scene path from config/cfg_args."""
    explicit_scene = config.get("camera_path", {}).get("orientation_scene")
    if explicit_scene:
        return explicit_scene

    bg_ply_path = config.get("background", {}).get("ply_path")
    if not bg_ply_path:
        return None

    search_dir = os.path.abspath(os.path.dirname(bg_ply_path))
    while True:
        cfg_args_path = os.path.join(search_dir, "cfg_args")
        source_path = parse_source_path_from_cfg_args(cfg_args_path)
        if source_path:
            return source_path
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir
    return None


def filter_gaussians(
    gs_data: Dict[str, np.ndarray],
    opacity_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    label: str = "gaussians",
) -> Dict[str, np.ndarray]:
    """Drop low-confidence or overly large gaussians that tend to produce floaters."""
    n = gs_data["xyz"].shape[0]
    mask = np.ones(n, dtype=bool)

    if opacity_min is not None:
        opacity = 1.0 / (1.0 + np.exp(-gs_data["opacity"].reshape(-1)))
        mask &= opacity >= float(opacity_min)

    if scale_max is not None:
        scale_world = np.exp(gs_data["scaling"])
        mask &= scale_world.max(axis=1) <= float(scale_max)

    kept = int(mask.sum())
    if kept == n:
        return gs_data

    print(f"[merge] Filtered {label}: kept {kept}/{n}")
    return {
        "xyz": gs_data["xyz"][mask],
        "features_dc": gs_data["features_dc"][mask],
        "opacity": gs_data["opacity"][mask],
        "scaling": gs_data["scaling"][mask],
        "rotation": gs_data["rotation"][mask],
    }


def cut_background_around_objects(
    bg_data: Dict[str, np.ndarray],
    cutouts: list[tuple[np.ndarray, float, float, float]],
    up: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Remove local background splats that would occlude inserted objects."""
    if not cutouts:
        return bg_data

    xyz = bg_data["xyz"].astype(np.float32, copy=False)
    up = up / np.linalg.norm(up)
    keep = np.ones(xyz.shape[0], dtype=bool)

    for center, radius, below, above in cutouts:
        rel = xyz - center.reshape(1, 3)
        height = rel @ up
        radial = rel - np.outer(height, up)
        radial_dist = np.linalg.norm(radial, axis=1)
        remove = (
            (radial_dist <= float(radius))
            & (height >= -float(below))
            & (height <= float(above))
        )
        keep &= ~remove
        print(
            "  background cutout:"
            f" center={np.round(center, 4).tolist()}"
            f" radius={float(radius):.3f}"
            f" removed={int(remove.sum())}"
        )

    if int(keep.sum()) == xyz.shape[0]:
        return bg_data

    print(f"[merge] Background cutouts: kept {int(keep.sum())}/{xyz.shape[0]}")
    return {
        "xyz": bg_data["xyz"][keep],
        "features_dc": bg_data["features_dc"][keep],
        "opacity": bg_data["opacity"][keep],
        "scaling": bg_data["scaling"][keep],
        "rotation": bg_data["rotation"][keep],
    }


def _perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / np.linalg.norm(direction)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(seed.dot(direction))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tangent_a = seed - seed.dot(direction) * direction
    tangent_a = tangent_a / np.linalg.norm(tangent_a)
    tangent_b = np.cross(direction, tangent_a)
    tangent_b = tangent_b / np.linalg.norm(tangent_b)
    return tangent_a.astype(np.float32), tangent_b.astype(np.float32)


def compute_anchor_point(
    xyz: np.ndarray,
    anchor_mode: str,
    rotation_deg: Optional[Tuple[float, float, float]] = None,
    scene_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the local-space anchor that should map to `translation`.

    `bottom_center` makes placement behave like "put this object down here".
    """
    anchor_mode = (anchor_mode or "none").lower()
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    if anchor_mode == "none":
        return np.zeros(3, dtype=np.float32)
    if anchor_mode == "bbox_center":
        return ((mins + maxs) * 0.5).astype(np.float32)
    if anchor_mode == "bottom_center" and scene_up is not None and rotation_deg is not None:
        rot = R.from_euler("xyz", np.radians(rotation_deg)).as_matrix()
        local_up = rot.T @ (scene_up / np.linalg.norm(scene_up))
        local_up = local_up.astype(np.float32)
        tangent_a, tangent_b = _perpendicular_basis(local_up)

        up_proj = xyz @ local_up
        a_proj = xyz @ tangent_a
        b_proj = xyz @ tangent_b
        anchor = (
            float(np.quantile(up_proj, 0.01)) * local_up
            + 0.5 * (float(a_proj.min()) + float(a_proj.max())) * tangent_a
            + 0.5 * (float(b_proj.min()) + float(b_proj.max())) * tangent_b
        )
        return anchor.astype(np.float32)
    if anchor_mode == "bottom_center":
        return np.array(
            [(mins[0] + maxs[0]) * 0.5, mins[1], (mins[2] + maxs[2]) * 0.5],
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")


def fit_local_support_plane(
    bg_xyz: np.ndarray,
    anchor_point: np.ndarray,
    up: np.ndarray,
    search_radius: float = 0.55,
    band_half_width: float = 0.05,
    min_points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the dominant horizontal support plane near an object anchor.

    The table top is typically the densest near-horizontal band in the upper
    half of the local neighborhood.
    """
    rel = bg_xyz - anchor_point[None, :]
    dist = np.linalg.norm(rel, axis=1)
    nearby = bg_xyz[dist <= search_radius]
    if nearby.shape[0] < min_points:
        raise ValueError(
            f"Not enough background points near anchor {anchor_point.tolist()} "
            f"(found {nearby.shape[0]}, need {min_points})."
        )

    up = up / np.linalg.norm(up)
    proj = nearby @ up
    lower = np.quantile(proj, 0.4)
    candidate = proj[proj >= lower]
    if candidate.shape[0] < min_points:
        candidate = proj

    bin_width = max(band_half_width * 0.8, 0.02)
    bins = np.arange(candidate.min(), candidate.max() + bin_width, bin_width)
    if bins.shape[0] < 2:
        bins = np.array([candidate.min(), candidate.max() + 1e-3], dtype=np.float64)
    hist, edges = np.histogram(candidate, bins=bins)
    peak_idx = int(np.argmax(hist))
    peak_center = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

    band_mask = np.abs(proj - peak_center) <= band_half_width
    band = nearby[band_mask]
    if band.shape[0] < min_points:
        raise ValueError(
            f"Support band near anchor {anchor_point.tolist()} is too sparse "
            f"(found {band.shape[0]}, need {min_points})."
        )

    mean = band.mean(axis=0)
    _, _, vh = np.linalg.svd(band - mean, full_matrices=False)
    normal = vh[-1]
    if normal.dot(up) < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    if normal.dot(up) < 0.8:
        normal = up

    return mean.astype(np.float32), normal.astype(np.float32)


def project_point_to_plane(
    point: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    clearance: float = 0.0,
) -> np.ndarray:
    """Project a point onto a plane and optionally lift it by `clearance`."""
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    signed_dist = np.dot(point - plane_point, plane_normal)
    return point - signed_dist * plane_normal + clearance * plane_normal


def transform_gaussians(
    gs_data: Dict[str, np.ndarray],
    T: np.ndarray,
    anchor: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply a 4×4 affine transform to gaussian splats.

    - xyz: transformed by T (rotation + scale + translation)
    - rotation: rotated by T's rotation component
    - scaling: scaled by T's scale component
    - features_dc, opacity: unchanged
    """
    xyz = gs_data["xyz"]
    n = xyz.shape[0]
    if anchor is None:
        anchor = np.zeros(3, dtype=np.float32)

    # Transform positions
    xyz_local = xyz - anchor.reshape(1, 3)
    xyz_h = np.hstack([xyz_local, np.ones((n, 1), dtype=np.float32)])
    xyz_new = (T @ xyz_h.T).T[:, :3]

    # Extract rotation + scale components
    R_mat = T[:3, :3]
    s = np.cbrt(np.abs(np.linalg.det(R_mat)))
    R_pure = R_mat / s if s > 1e-8 else R_mat

    # Transform each gaussian's rotation quaternion. Stored values are raw
    # parameters in GS checkpoints, but normalizing before composition matches
    # the renderer's activation.
    old_rot = gs_data["rotation"]  # (N, 4) — quat (w, x, y, z)
    old_rot = old_rot / np.clip(np.linalg.norm(old_rot, axis=1, keepdims=True), 1e-8, None)
    old_R_mats = R.from_quat(old_rot[:, [1, 2, 3, 0]]).as_matrix()  # scipy uses (x,y,z,w)
    new_R_mats = R_pure @ old_R_mats
    new_quats = R.from_matrix(new_R_mats).as_quat()  # (N, 4) (x,y,z,w)
    # Convert back to (w,x,y,z)
    new_rotation = np.zeros_like(old_rot)
    new_rotation[:, 0] = new_quats[:, 3]
    new_rotation[:, 1] = new_quats[:, 0]
    new_rotation[:, 2] = new_quats[:, 1]
    new_rotation[:, 3] = new_quats[:, 2]

    # Scale is stored in log-space in Gaussian Splatting checkpoints, so a
    # world-space uniform scale composes additively in raw parameter space.
    new_scaling = gs_data["scaling"] + np.log(max(s, 1e-8))

    return {
        "xyz": xyz_new.astype(np.float32),
        "features_dc": gs_data["features_dc"],
        "opacity": gs_data["opacity"],
        "scaling": new_scaling.astype(np.float32),
        "rotation": new_rotation.astype(np.float32),
    }


def merge_gaussians(*gs_list: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Concatenate multiple gaussian dicts into one."""
    if len(gs_list) == 0:
        raise ValueError("No gaussians to merge")
    if len(gs_list) == 1:
        return gs_list[0]

    return {
        "xyz": np.concatenate([g["xyz"] for g in gs_list], axis=0),
        "features_dc": np.concatenate([g["features_dc"] for g in gs_list], axis=0),
        "opacity": np.concatenate([g["opacity"] for g in gs_list], axis=0),
        "scaling": np.concatenate([g["scaling"] for g in gs_list], axis=0),
        "rotation": np.concatenate([g["rotation"] for g in gs_list], axis=0),
    }


def parse_pose(pose_str: str):
    """Parse 'tx,ty,tz,scale,rx,ry,rz' into tuple of floats."""
    vals = [float(v) for v in pose_str.split(",")]
    assert len(vals) == 7, "pose must have 7 values: tx,ty,tz,scale,rx,ry,rz"
    return vals


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple GS PLY files into one"
    )
    parser.add_argument("--config", default="configs/fusion.yaml",
                        help="YAML config with object paths and poses")
    parser.add_argument("--bg", default=None, help="Background PLY path")
    parser.add_argument("--obj_a", default=None)
    parser.add_argument("--pose_a", default=None)
    parser.add_argument("--obj_b", default=None)
    parser.add_argument("--pose_b", default=None)
    parser.add_argument("--obj_c", default=None)
    parser.add_argument("--pose_c", default=None)
    parser.add_argument("--output", default="outputs/merged.ply")
    args = parser.parse_args()

    # Load config if provided
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Resolve paths and poses
    def _resolve(obj_key, obj_arg, pose_arg):
        if obj_arg and pose_arg:
            return obj_arg, parse_pose(pose_arg), {}
        entry = config.get("objects", {}).get(obj_key, {})
        if entry:
            return entry["ply_path"], (
                entry["translation"] + [entry["scale"]] + entry["rotation"]
            ), entry
        return None, None, {}

    parts = []
    background_xyz = None
    bg_data = None
    background_cutouts = []
    scene_up = None
    orientation_scene = infer_orientation_scene_path(config)
    if orientation_scene:
        try:
            orientation = estimate_orientation_from_colmap(orientation_scene)
            scene_up = orientation["up"].astype(np.float32)
            print(f"[merge] Estimated scene up from COLMAP: {np.round(scene_up, 4).tolist()}")
        except Exception as exc:
            print(f"[merge] WARNING: failed to estimate scene orientation: {exc}")

    # Load background (no transform — it defines the world frame)
    bg_path = args.bg or config.get("background", {}).get("ply_path")
    if bg_path and os.path.exists(bg_path):
        print(f"[merge] Loading background: {bg_path}")
        bg_data = load_gaussian_ply(bg_path)
        bg_filter = config.get("background", {}).get("filter", {})
        if bg_filter:
            bg_data = filter_gaussians(
                bg_data,
                opacity_min=bg_filter.get("opacity_min"),
                scale_max=bg_filter.get("scale_max"),
            )
        print(f"  {bg_data['xyz'].shape[0]} gaussians")
        background_xyz = bg_data["xyz"].astype(np.float32)
    else:
        print("[merge] WARNING: No background loaded.")

    # Load and transform each object
    for key in ["object_a", "object_b", "object_c"]:
        obj_path, pose, entry = _resolve(key, getattr(args, f"obj_{key[-1]}", None),
                                         getattr(args, f"pose_{key[-1]}", None))
        if obj_path and os.path.exists(obj_path) and pose:
            print(f"[merge] Loading {key}: {obj_path}")
            obj_data = load_gaussian_ply(obj_path)
            print(f"  {obj_data['xyz'].shape[0]} gaussians → pose: {pose}")
            obj_filter = entry.get("filter", {})
            if obj_filter:
                obj_data = filter_gaussians(
                    obj_data,
                    opacity_min=obj_filter.get("opacity_min"),
                    scale_max=obj_filter.get("scale_max"),
                    label=key,
                )
            anchor_mode = entry.get("anchor_mode", "none")
            anchor = compute_anchor_point(
                obj_data["xyz"],
                anchor_mode,
                rotation_deg=tuple(pose[4:]),
                scene_up=scene_up,
            )

            translation = np.array(pose[:3], dtype=np.float32)
            if entry.get("ground_to_surface", False) and background_xyz is not None and scene_up is not None:
                plane_point, plane_normal = fit_local_support_plane(
                    background_xyz,
                    translation,
                    scene_up,
                    search_radius=float(entry.get("support_radius", 0.55)),
                    band_half_width=float(entry.get("support_band_half_width", 0.05)),
                )
                translation = project_point_to_plane(
                    translation,
                    plane_point,
                    plane_normal,
                    clearance=float(entry.get("clearance", 0.0)),
                )
                print(
                    "  support plane:"
                    f" point={np.round(plane_point, 4).tolist()}"
                    f" normal={np.round(plane_normal, 4).tolist()}"
                    f" grounded_translation={np.round(translation, 4).tolist()}"
                )

            cutout_radius = entry.get("background_cutout_radius")
            if cutout_radius is not None and scene_up is not None:
                background_cutouts.append(
                    (
                        translation.copy(),
                        float(cutout_radius),
                        float(entry.get("background_cutout_below", 0.05)),
                        float(entry.get("background_cutout_above", 0.45)),
                    )
                )

            T = build_transform_matrix(tuple(translation.tolist()), pose[3], pose[4:])
            obj_data = transform_gaussians(obj_data, T, anchor=anchor)
            parts.append(obj_data)
        elif obj_path:
            print(f"[merge] WARNING: {key} PLY not found: {obj_path}")

    foreground_ply = config.get("output", {}).get("foreground_ply")
    if foreground_ply and parts:
        foreground = merge_gaussians(*parts)
        save_gaussian_ply(foreground, foreground_ply)
        print(f"[merge] Saved foreground objects to {foreground_ply}")

    if bg_data is not None:
        if scene_up is not None:
            bg_data = cut_background_around_objects(bg_data, background_cutouts, scene_up)
        parts.insert(0, bg_data)

    if len(parts) < 1:
        print("[merge] ERROR: No objects to merge.")
        sys.exit(1)

    merged = merge_gaussians(*parts)
    total = merged["xyz"].shape[0]
    print(f"[merge] Total merged gaussians: {total}")

    save_gaussian_ply(merged, args.output)
    print(f"[merge] Saved to {args.output}")


if __name__ == "__main__":
    main()
