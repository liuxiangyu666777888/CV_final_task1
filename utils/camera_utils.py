"""
camera_utils.py — Camera trajectory generation for multi-view rendering.

Generates camera poses (world-to-camera matrices) along smooth paths:
  - spiral: helical path around a look-at point
  - circle: planar circle at fixed height

Also provides scene-orientation utilities that estimate a stable world `up`
direction from COLMAP/SfM camera poses. This matters for real captures and
2DGS scenes because the reconstructed world frame is not guaranteed to align
with the canonical y-up axis.
"""

import importlib.util
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a vector and keep the original value if its norm is tiny."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.copy()
    return vec / norm


def _project_to_plane(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project a vector onto the plane with the given unit normal."""
    return vec - np.dot(vec, normal) * normal


def _fallback_perpendicular_axis(normal: np.ndarray) -> np.ndarray:
    """Choose a stable axis perpendicular to `normal`."""
    candidates = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]
    for candidate in candidates:
        axis = _project_to_plane(candidate, normal)
        if np.linalg.norm(axis) > 1e-6:
            return _normalize(axis)
    raise ValueError("Cannot build a perpendicular axis for the provided normal.")


def _build_orbit_frame(
    up: np.ndarray,
    start_dir: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis for an orbit plane.

    Returns:
        tangent_x, tangent_z such that both are perpendicular to `up`.
    """
    up = _normalize(up)
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("Orbit frame requires a non-zero up vector.")

    if start_dir is not None:
        tangent_x = _project_to_plane(np.asarray(start_dir, dtype=np.float32), up)
        if np.linalg.norm(tangent_x) > 1e-6:
            tangent_x = _normalize(tangent_x)
        else:
            tangent_x = _fallback_perpendicular_axis(up)
    else:
        tangent_x = _fallback_perpendicular_axis(up)

    tangent_z = _normalize(np.cross(up, tangent_x))
    if np.linalg.norm(tangent_z) < 1e-6:
        tangent_x = _fallback_perpendicular_axis(up)
        tangent_z = _normalize(np.cross(up, tangent_x))
    return tangent_x, tangent_z


def _load_colmap_loader_module():
    """Load the COLMAP parser without importing the full 2DGS package."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loader_path = os.path.join(
        repo_root, "2d-gaussian-splatting", "scene", "colmap_loader.py"
    )
    if not os.path.exists(loader_path):
        raise FileNotFoundError(
            f"COLMAP loader not found at {loader_path}. "
            "Expected the bundled 2d-gaussian-splatting repo."
        )

    spec = importlib.util.spec_from_file_location("camera_utils_colmap_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load COLMAP utilities from {loader_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate_orientation_from_colmap(
    scene_path: str,
) -> Dict[str, np.ndarray]:
    """
    Estimate a stable scene orientation from COLMAP camera poses.

    The world frame from SfM is arbitrary. For roaming videos we infer:
      - `up`: average camera-up direction
      - `focus_point`: least-squares intersection of camera viewing rays
      - `orbit_radius_mean`: typical camera distance to the focus in the orbit plane
      - `height_*`: camera heights along the inferred up axis

    Args:
        scene_path: dataset root that contains `sparse/0/images.bin` or `.txt`

    Returns:
        Dict containing orientation and camera-layout diagnostics.
    """
    colmap_loader = _load_colmap_loader_module()

    images_bin = os.path.join(scene_path, "sparse", "0", "images.bin")
    images_txt = os.path.join(scene_path, "sparse", "0", "images.txt")
    if os.path.exists(images_bin):
        images = colmap_loader.read_extrinsics_binary(images_bin)
    elif os.path.exists(images_txt):
        images = colmap_loader.read_extrinsics_text(images_txt)
    else:
        raise FileNotFoundError(
            f"No COLMAP extrinsics found under {scene_path}/sparse/0"
        )

    centers = []
    ups = []
    forwards = []
    for image in images.values():
        world_to_camera = colmap_loader.qvec2rotmat(image.qvec)
        translation = np.asarray(image.tvec, dtype=np.float32)
        camera_to_world = world_to_camera.T
        center = -(camera_to_world @ translation)
        centers.append(center)
        ups.append(_normalize(camera_to_world[:, 1]))
        forwards.append(_normalize(-camera_to_world[:, 2]))

    centers = np.stack(centers).astype(np.float32)
    ups = np.stack(ups).astype(np.float32)
    forwards = np.stack(forwards).astype(np.float32)

    mean_up = _normalize(ups.mean(axis=0))
    if np.linalg.norm(mean_up) < 1e-6:
        raise ValueError("Failed to estimate a stable up vector from COLMAP cameras.")

    # A least-squares line intersection is more useful than the raw mean camera
    # center when the training views were captured around an object/region.
    proj_accum = np.zeros((3, 3), dtype=np.float64)
    rhs_accum = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for center, forward in zip(centers, forwards):
        direction = _normalize(forward.astype(np.float64))
        proj = identity - np.outer(direction, direction)
        proj_accum += proj
        rhs_accum += proj @ center.astype(np.float64)
    focus_point = np.linalg.lstsq(proj_accum, rhs_accum, rcond=None)[0].astype(np.float32)

    centered = centers - centers.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = vh[-1].astype(np.float32)
    if np.dot(plane_normal, mean_up) < 0:
        plane_normal = -plane_normal
    plane_normal = _normalize(plane_normal)

    heights = (centers - focus_point[None, :]) @ plane_normal
    radial = (centers - focus_point[None, :]) - np.outer(heights, plane_normal)
    radii = np.linalg.norm(radial, axis=1)

    return {
        "up": plane_normal.astype(np.float32),
        "mean_camera_up": mean_up.astype(np.float32),
        "focus_point": focus_point.astype(np.float32),
        "camera_centers": centers.astype(np.float32),
        "camera_forwards": forwards.astype(np.float32),
        "orbit_radius_mean": np.array(radii.mean(), dtype=np.float32),
        "orbit_radius_min": np.array(radii.min(), dtype=np.float32),
        "orbit_radius_max": np.array(radii.max(), dtype=np.float32),
        "height_min": np.array(heights.min(), dtype=np.float32),
        "height_max": np.array(heights.max(), dtype=np.float32),
        "height_mean": np.array(heights.mean(), dtype=np.float32),
        "height_std": np.array(heights.std(), dtype=np.float32),
        "singular_values": singular_values.astype(np.float32),
    }


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build a 4×4 world-to-camera (view) matrix from eye/center/up.

    Args:
        eye:    (3,) camera position in world
        center: (3,) look-at target
        up:     (3,) world up direction

    Returns:
        (4, 4) view matrix (world-to-camera)
    """
    z = _normalize(eye - center)
    x = np.cross(_normalize(up), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(_fallback_perpendicular_axis(z), z)
    x = _normalize(x)
    y = _normalize(np.cross(z, x))

    rot = np.eye(4, dtype=np.float32)
    rot[0, :3] = x
    rot[1, :3] = y
    rot[2, :3] = z

    trans = np.eye(4, dtype=np.float32)
    trans[:3, 3] = -eye

    return rot @ trans


def generate_spiral_path(
    num_frames: int = 300,
    radius: float = 3.0,
    height_start: float = 0.5,
    height_end: float = 2.0,
    turns: float = 2.0,
    angle_offset_deg: float = 0.0,
    look_at: Tuple[float, float, float] = (0.0, 0.5, 0.0),
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    start_dir: Optional[Tuple[float, float, float]] = None,
) -> List[np.ndarray]:
    """
    Generate a spiral-rising camera trajectory.

    Args:
        num_frames:   total frame count
        radius:       spiral radius in world units
        height_start: initial camera height
        height_end:   final camera height
        turns:        number of full rotations
        angle_offset_deg: starting azimuth offset in degrees
        look_at:      (x,y,z) focal point
        up:           world up vector

    Returns:
        list of (4,4) view matrices, one per frame
    """
    center = np.array(look_at, dtype=np.float32)
    up = _normalize(np.array(up, dtype=np.float32))
    orbit_x, orbit_z = _build_orbit_frame(
        up, None if start_dir is None else np.array(start_dir, dtype=np.float32)
    )
    angle_offset = math.radians(angle_offset_deg)

    matrices = []
    for i in range(num_frames):
        t = 0.0 if num_frames <= 1 else i / (num_frames - 1)
        angle = angle_offset + t * turns * 2.0 * math.pi
        height = height_start + t * (height_end - height_start)

        eye = (
            center
            + radius * math.cos(angle) * orbit_x
            + radius * math.sin(angle) * orbit_z
            + height * up
        ).astype(np.float32)

        view = _look_at(eye, center, up)
        matrices.append(view)

    return matrices


def generate_circle_path(
    num_frames: int = 300,
    radius: float = 3.0,
    height: float = 1.0,
    angle_offset_deg: float = 0.0,
    sweep_deg: float = 360.0,
    look_at: Tuple[float, float, float] = (0.0, 0.5, 0.0),
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    start_dir: Optional[Tuple[float, float, float]] = None,
) -> List[np.ndarray]:
    """
    Generate a flat circular camera trajectory at fixed height.

    Args:
        num_frames: total frame count
        radius:     circle radius
        height:     fixed camera height
        angle_offset_deg: starting azimuth offset in degrees
        sweep_deg: total azimuth sweep in degrees
        look_at:    (x,y,z) focal point

    Returns:
        list of (4,4) view matrices
    """
    center = np.array(look_at, dtype=np.float32)
    up = _normalize(np.array(up, dtype=np.float32))
    orbit_x, orbit_z = _build_orbit_frame(
        up, None if start_dir is None else np.array(start_dir, dtype=np.float32)
    )
    angle_offset = math.radians(angle_offset_deg)
    sweep = math.radians(sweep_deg)

    matrices = []
    for i in range(num_frames):
        t = 0.0 if num_frames <= 1 else i / (num_frames - 1)
        angle = angle_offset + t * sweep
        eye = (
            center
            + radius * math.cos(angle) * orbit_x
            + radius * math.sin(angle) * orbit_z
            + height * up
        ).astype(np.float32)
        matrices.append(_look_at(eye, center, up))

    return matrices


def get_projection_matrix(
    fov_y: float = 49.1,
    aspect: float = 1920.0 / 1080.0,
    near: float = 0.01,
    far: float = 100.0,
) -> np.ndarray:
    """
    Build a 4×4 OpenGL-style perspective projection matrix.

    Args:
        fov_y:  vertical field-of-view in degrees
        aspect: width / height
        near:   near plane
        far:    far plane

    Returns:
        (4, 4) projection matrix
    """
    f = 1.0 / math.tan(math.radians(fov_y) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj
