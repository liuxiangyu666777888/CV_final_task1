"""
io_utils.py — PLY read/write utilities for 2D Gaussian Splatting format.

Handles the standard Gaussian Splatting PLY format with attributes:
  x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity,
  scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
"""

import numpy as np
from plyfile import PlyData, PlyElement
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# PLY I/O
# ---------------------------------------------------------------------------

def _ply_field_names(vert) -> set:
    """Return the set of field names in a PLY vertex element."""
    if hasattr(vert, "data") and getattr(vert.data, "dtype", None) is not None:
        return set(vert.data.dtype.names)
    return set(vert.dtype().names)


def _rotation_from_normals(normals: np.ndarray) -> np.ndarray:
    """
    Convert surface normals (nx, ny, nz) to quaternions (w, x, y, z)
    that rotate the z-axis to align with each normal.
    """
    n = normals.shape[0]
    default_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rotations = []
    for i in range(n):
        ni = normals[i] / (np.linalg.norm(normals[i]) + 1e-8)
        v = np.cross(default_z, ni)
        c = np.dot(default_z, ni)
        if c < -0.9999:
            rot = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        else:
            s = np.sqrt((1 + c) * 2)
            inv_s = 1.0 / s
            rot = np.array([s * 0.5, v[0] * inv_s, v[1] * inv_s, v[2] * inv_s], dtype=np.float32)
        rotations.append(rot)
    return np.stack(rotations, axis=0).astype(np.float32)


def load_gaussian_ply(ply_path: str) -> Dict[str, np.ndarray]:
    """
    Load a 2DGS or standard 3DGS .ply file.  Auto-detects the format.

    2DGS format (from the paper): x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..N
    Standard 3DGS format (gsplat): x, y, z, f_dc_0..2, opacity, scale_0..2, rot_0..3

    Returns a dict with keys:
        xyz:         (N, 3)  float32 — 3D positions
        features_dc: (N, 3, 1) float32 — SH DC colour
        opacity:     (N, 1)  float32 — opacity (raw value)
        scaling:     (N, 3)  float32 — log-scales
        rotation:    (N, 4)  float32 — quaternion (w, x, y, z)
    """
    plydata = PlyData.read(ply_path)
    vert = plydata["vertex"]
    fields = _ply_field_names(vert)
    data = vert.data if hasattr(vert, "data") else vert
    n = vert["x"].shape[0]

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)

    # SH DC colours (present in both formats)
    features_dc = np.zeros((n, 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = data["f_dc_0"]
    features_dc[:, 1, 0] = data["f_dc_1"]
    features_dc[:, 2, 0] = data["f_dc_2"]

    # --- Format detection ---
    if "opacity" in fields and {"rot_0", "rot_1", "rot_2", "rot_3"} <= fields:
        opacity = data["opacity"].astype(np.float32).reshape(-1, 1)
        rotation = np.stack(
            [data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]],
            axis=-1,
        ).astype(np.float32)

        if "scale_2" in fields:
            # Standard 3DGS format — read all three scales directly.
            scaling = np.stack(
                [data["scale_0"], data["scale_1"], data["scale_2"]],
                axis=-1,
            ).astype(np.float32)
        elif {"scale_0", "scale_1"} <= fields:
            # 2DGS stores two in-plane scales. Synthesize a third scale so downstream
            # scripts can treat it as a regular 3D Gaussian.
            scale_0 = data["scale_0"].astype(np.float32)
            scale_1 = data["scale_1"].astype(np.float32)
            scale_2 = np.minimum(scale_0, scale_1)
            scaling = np.stack([scale_0, scale_1, scale_2], axis=-1)
        else:
            raise KeyError(f"PLY {ply_path} has opacity/rotation but no recognized scale fields: {sorted(fields)}")
    else:
        # 2DGS format — derive missing fields from available data
        opacity = np.full((n, 1), 0.9, dtype=np.float32)

        # Derive scale from nearest neighbour distances
        from scipy.spatial import KDTree
        tree = KDTree(xyz)
        dists, _ = tree.query(xyz, k=min(10, n))
        avg_nn_dist = dists[:, 1:].mean(axis=1)
        scale_val = np.clip(avg_nn_dist * 0.5, 0.001, 0.1)
        scaling = np.stack([scale_val, scale_val, scale_val], axis=-1).astype(np.float32)

        # Derive rotation from normals if available, otherwise identity
        if "nx" in fields:
            normals = np.stack([data["nx"], data["ny"], data["nz"]], axis=-1).astype(np.float32)
            rotation = _rotation_from_normals(normals)
        else:
            rotation = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1))

    return {
        "xyz": xyz,
        "features_dc": features_dc,
        "opacity": opacity,
        "scaling": scaling,
        "rotation": rotation,
    }


def save_gaussian_ply(gs_data: Dict[str, np.ndarray], output_path: str) -> None:
    """
    Save a gaussian splat dict to a standard .ply file.

    Accepts the same dict layout returned by load_gaussian_ply().
    """

    xyz = gs_data["xyz"].astype(np.float32)
    n = xyz.shape[0]

    # Ensure shapes
    features_dc = gs_data["features_dc"]
    if features_dc.ndim == 3:
        features_dc = features_dc.reshape(n, 3)

    opacity = gs_data["opacity"].reshape(n).astype(np.float32)
    scaling = gs_data["scaling"].astype(np.float32)
    rotation = gs_data["rotation"].astype(np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    vertices = np.empty(n, dtype=dtype)
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["f_dc_0"] = features_dc[:, 0]
    vertices["f_dc_1"] = features_dc[:, 1]
    vertices["f_dc_2"] = features_dc[:, 2]
    vertices["opacity"] = opacity
    vertices["scale_0"] = scaling[:, 0]
    vertices["scale_1"] = scaling[:, 1]
    vertices["scale_2"] = scaling[:, 2]
    vertices["rot_0"] = rotation[:, 0]
    vertices["rot_1"] = rotation[:, 1]
    vertices["rot_2"] = rotation[:, 2]
    vertices["rot_3"] = rotation[:, 3]

    el = PlyElement.describe(vertices, "vertex")
    PlyData([el]).write(output_path)
    print(f"[io_utils] Saved {n} gaussians to {output_path}")


# ---------------------------------------------------------------------------
# COLMAP camera helpers
# ---------------------------------------------------------------------------

def read_colmap_cameras_text(cam_path: str) -> dict:
    """
    Read COLMAP cameras.txt (text format).

    Returns:
        dict[camera_id] -> {
            "model": str,
            "width": int, "height": int,
            "params": list[float],
        }
    """
    cameras = {}
    with open(cam_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_colmap_images_text(img_path: str) -> dict:
    """
    Read COLMAP images.txt (text format).

    Returns:
        dict[image_id] -> {
            "qvec": np.ndarray (4,),  # quaternion (w, x, y, z)
            "tvec": np.ndarray (3,),  # translation
            "camera_id": int,
            "name": str,
        }
    """
    images = {}
    with open(img_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        images[img_id] = {
            "qvec": np.array([qw, qx, qy, qz], dtype=np.float32),
            "tvec": np.array([tx, ty, tz], dtype=np.float32),
            "camera_id": camera_id,
            "name": name,
        }
        i += 2  # skip points line
    return images


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float32)
