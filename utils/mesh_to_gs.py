"""
mesh_to_gs.py — Convert textured Mesh to 2D Gaussian Splatting PLY format.

Algorithm:
  1. Load mesh via trimesh
  2. Sample N points on surface (area-weighted triangle sampling)
  3. Extract colour from texture map (or use vertex colours / white)
  4. Compute per-point covariance from face normal + face area
  5. Export as standard 2DGS .ply
"""

import numpy as np
import trimesh
from typing import Dict, Optional, Tuple


def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    num_samples: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points uniformly on mesh surface (area-weighted).

    Args:
        mesh:        trimesh mesh object
        num_samples: target number of points

    Returns:
        positions:    (N, 3) point positions
        normals:      (N, 3) surface normals at each point
        colors:       (N, 3) RGB colours (0..1) from texture or vertex colour
        face_indices: (N,)  index of the face each point belongs to
    """
    # Area-weighted sampling. Keep the same barycentric coordinates for both
    # positions and UVs so texture colours match the sampled surface points.
    if len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces to sample")

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    valid_faces = np.isfinite(face_areas) & (face_areas > 0)
    if not np.any(valid_faces):
        raise ValueError("Mesh has no positive-area faces to sample")

    probs = np.zeros_like(face_areas, dtype=np.float64)
    probs[valid_faces] = face_areas[valid_faces]
    probs /= probs.sum()
    face_indices = np.random.choice(len(mesh.faces), size=num_samples, p=probs)

    r1 = np.random.rand(num_samples, 1)
    r2 = np.random.rand(num_samples, 1)
    sqrt_r1 = np.sqrt(r1)
    bary = np.concatenate(
        [
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ],
        axis=1,
    ).astype(np.float32)

    triangles = mesh.triangles[face_indices]
    points = (triangles * bary[:, :, None]).sum(axis=1)

    # Per-point normals
    normals = mesh.face_normals[face_indices]

    # Per-point colours
    if mesh.visual.kind == "texture" and hasattr(mesh.visual, "uv"):
        # Sample from texture image
        try:
            uv = mesh.visual.uv
            face_uvs = uv[mesh.faces[face_indices]]   # (N, 3, 2)
            interp_uv = (face_uvs * bary[:, :, None]).sum(axis=1)
            tex = mesh.visual.material.image  # PIL Image
            tex_np = np.array(tex.convert("RGBA")).astype(np.float32) / 255.0
            h, w_tex = tex_np.shape[:2]
            interp_uv = np.mod(interp_uv, 1.0)
            px = np.clip((interp_uv[:, 0] * (w_tex - 1)).astype(int), 0, w_tex - 1)
            py = np.clip(((1.0 - interp_uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
            colors = tex_np[py, px, :3]
        except Exception:
            colors = np.full((num_samples, 3), 0.8, dtype=np.float32)
    elif mesh.visual.kind == "vertex" and hasattr(mesh.visual, "vertex_colors"):
        # Interpolate vertex colours
        vc = mesh.visual.vertex_colors.astype(np.float32)[:, :3] / 255.0
        faces = mesh.faces[face_indices]
        colors = (vc[faces] * bary[:, :, None]).sum(axis=1)
    else:
        # Default: white
        colors = np.full((num_samples, 3), 0.8, dtype=np.float32)

    return points, normals, colors, face_indices


def mesh_to_gaussian_splats(
    mesh_path: str,
    num_samples: int = 50000,
    texture_path: Optional[str] = None,
    base_opacity: float = 0.9,
    scale_factor: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Convert a mesh file to Gaussian Splatting parameters.

    Args:
        mesh_path:     path to .obj / .ply mesh
        num_samples:   number of gaussian splats
        texture_path:  optional texture image override
        base_opacity:  initial opacity (sigmoid-inverse space value, e.g. 0.9)
        scale_factor:  controls initial splat size relative to local face area

    Returns:
        dict with keys: xyz, features_dc, opacity, scaling, rotation
    """
    # Load mesh
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        # If multi-mesh scene, merge
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry found in {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)

    # If explicit texture path provided, apply it
    if texture_path is not None:
        from PIL import Image
        tex_img = Image.open(texture_path)
        uv = mesh.visual.uv if hasattr(mesh.visual, "uv") else None
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=tex_img)

    # Sample surface — face_indices reused for scaling computation
    positions, normals, colors, face_indices = sample_mesh_surface(mesh, num_samples)

    # ---- Build Gaussian parameters ----

    # xyz
    xyz = positions.astype(np.float32)

    # features_dc — store raw SH DC coefficients, matching 2DGS/3DGS.
    c0 = 0.28209479177387814
    features_dc = ((colors.astype(np.float32) - 0.5) / c0).reshape(-1, 3, 1)

    # opacity — store the pre-sigmoid raw parameter.
    alpha = np.clip(base_opacity, 1e-4, 1 - 1e-4)
    opacity = np.full((num_samples, 1), np.log(alpha / (1 - alpha)), dtype=np.float32)

    # scaling — derived from local face area
    # Use small isotropic scale initially, refining possible via short optim
    if len(mesh.faces) > 0:
        face_areas = trimesh.triangles.area(mesh.triangles[face_indices])
        avg_edge = np.sqrt(np.clip(face_areas, 1e-8, None)) * scale_factor
    else:
        avg_edge = np.full(num_samples, scale_factor, dtype=np.float32)
    scaling_linear = np.stack([
        avg_edge,
        avg_edge,
        avg_edge,
    ], axis=-1).astype(np.float32)
    # scaling — store the pre-exp raw parameter.
    scaling = np.log(np.clip(scaling_linear, 1e-8, None)).astype(np.float32)

    # rotation — align with surface normal
    # Each gaussian's rotation quaternion maps z-axis to surface normal
    default_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rotations = []
    for n in normals:
        n = n / (np.linalg.norm(n) + 1e-8)
        # Compute rotation from default_z to surface normal n
        v = np.cross(default_z, n)
        c = np.dot(default_z, n)
        if c < -0.9999:
            # Anti-parallel: 180-degree rotation around x-axis
            rot = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # quat (w,x,y,z)
        else:
            s = np.sqrt((1 + c) * 2)
            inv_s = 1.0 / s
            rot = np.array([
                s * 0.5,        # w
                v[0] * inv_s,   # x
                v[1] * inv_s,   # y
                v[2] * inv_s,   # z
            ], dtype=np.float32)
        rotations.append(rot)
    rotation = np.stack(rotations, axis=0).astype(np.float32)

    return {
        "xyz": xyz,
        "features_dc": features_dc,
        "opacity": opacity,
        "scaling": scaling,
        "rotation": rotation,
    }


def refine_gaussian_from_mesh(
    mesh_path: str,
    output_ply: str,
    num_samples: int = 50000,
) -> None:
    """
    High-level convenience: mesh -> gaussian splats -> save PLY.

    Args:
        mesh_path:   input mesh (.obj / .ply)
        output_ply:  output .ply path
        num_samples: splat count
    """
    from utils.io_utils import save_gaussian_ply

    gs_data = mesh_to_gaussian_splats(mesh_path, num_samples=num_samples)
    save_gaussian_ply(gs_data, output_ply)
