#!/usr/bin/env python
"""Export a plain OBJ mesh from a trained Magic123 DMTet checkpoint.

This bypasses Magic123's UV unwrap / texture baking path, which can crash in
native code after training has already finished.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict


PROJECT_DIR = Path(__file__).resolve().parents[2]
MAGIC123_DIR = PROJECT_DIR / "Magic123"
if str(MAGIC123_DIR) not in sys.path:
    sys.path.insert(0, str(MAGIC123_DIR))

from main import parser as magic_parser  # noqa: E402
from nerf.utils import seed_everything, setup_workspace  # noqa: E402


def load_project_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_opt(args: argparse.Namespace, project_cfg: dict) -> edict:
    base_args = magic_parser.parse_args([])
    opt = edict(vars(base_args))

    input_image = PROJECT_DIR / project_cfg["input_image"]
    text_prompt = project_cfg.get("text_prompt") or "A high-resolution DSLR image of an object"

    opt.O = True
    opt.fp16 = True
    opt.cuda_ray = True
    opt.workspace = str(args.workspace)
    opt.text = text_prompt
    opt.image = str(input_image)
    opt.dmtet = True
    opt.init_ckpt = str(args.stage1_ckpt) if args.stage1_ckpt else ""
    opt.tet_grid_size = int(project_cfg["stage2"]["tet_resolution"])
    opt.iters = int(project_cfg["stage2"]["iterations"])
    opt.lr = float(project_cfg["stage2"]["lr"])
    opt.guidance = ["SD"]
    opt.lambda_guidance = [1e-3]
    opt.guidance_scale = [float(project_cfg["stage1"]["guidance"]["guidance_scale"])]
    opt.known_view_interval = 4
    opt.latent_iter_ratio = 0
    opt.textureless_iter_ratio = 0
    opt.albedo_iter_ratio = 0
    opt.normal_iter_ratio = 0
    opt.progressive_view = False
    opt.progressive_level = False
    opt.rm_edge = True
    opt.bg_radius = -1
    opt.h = int(opt.h * opt.dmtet_reso_scale)
    opt.w = int(opt.w * opt.dmtet_reso_scale)
    opt.known_view_scale = 1
    opt.grid_levels_mask = -1
    opt.t_range = [0.02, 0.50]
    opt.lambda_normal = 0
    opt.lambda_depth = 0
    opt.default_zero123_w = 1

    opt.images = [opt.image]
    opt.ref_radii = [opt.default_radius]
    opt.ref_polars = [opt.default_polar]
    opt.ref_azimuths = [opt.default_azimuth]
    opt.zero123_ws = [opt.default_zero123_w]

    setup_workspace(opt)
    if opt.seed < 0:
        opt.seed = 101
    seed_everything(int(opt.seed))

    return opt


def make_model(opt: edict) -> torch.nn.Module:
    if opt.backbone == "vanilla":
        from nerf.network import NeRFNetwork
    elif opt.backbone == "grid":
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == "grid_tcnn":
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == "grid_taichi":
        raise NotImplementedError("grid_taichi export-only path is not supported here")
    else:
        raise NotImplementedError(f"Unsupported backbone: {opt.backbone}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    return NeRFNetwork(opt).to(device)


def sample_vertex_albedo(model: torch.nn.Module, vertices: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
    colors = []
    head = 0
    while head < vertices.shape[0]:
        tail = min(head + batch_size, vertices.shape[0])
        colors.append(model.density(vertices[head:tail])["albedo"].float().clamp(0, 1))
        head = tail
    return torch.cat(colors, dim=0)


def build_face_texture_atlas(
    faces: np.ndarray,
    face_colors: np.ndarray,
    tile_size: int = 8,
    tile_pad: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_faces = max(int(faces.shape[0]), 1)
    grid_cols = int(math.ceil(math.sqrt(num_faces)))
    grid_rows = int(math.ceil(num_faces / grid_cols))
    atlas_w = grid_cols * tile_size
    atlas_h = grid_rows * tile_size

    texture = np.zeros((atlas_h, atlas_w, 3), dtype=np.uint8)
    vt = np.zeros((num_faces * 3, 2), dtype=np.float32)
    ft = np.arange(num_faces * 3, dtype=np.int32).reshape(num_faces, 3)

    tile_pad = min(tile_pad, max(tile_size // 2 - 1, 0))
    x0_off = tile_pad + 0.5
    x1_off = tile_size - tile_pad - 0.5
    y0_off = tile_pad + 0.5
    y1_off = tile_size - tile_pad - 0.5
    if x1_off <= x0_off:
        x0_off, x1_off = 0.5, tile_size - 0.5
    if y1_off <= y0_off:
        y0_off, y1_off = 0.5, tile_size - 0.5

    for i in range(faces.shape[0]):
        row = i // grid_cols
        col = i % grid_cols
        px = col * tile_size
        py = row * tile_size
        texture[py:py + tile_size, px:px + tile_size] = face_colors[i]

        vt[3 * i + 0] = [(px + x0_off) / atlas_w, 1.0 - (py + y0_off) / atlas_h]
        vt[3 * i + 1] = [(px + x1_off) / atlas_w, 1.0 - (py + y0_off) / atlas_h]
        vt[3 * i + 2] = [(px + x0_off) / atlas_w, 1.0 - (py + y1_off) / atlas_h]

    return texture, vt, ft


def write_textured_obj(
    obj_path: Path,
    mtl_path: Path,
    texture_name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    vt: np.ndarray,
    ft: np.ndarray,
) -> None:
    with obj_path.open("w", encoding="ascii") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        for x, y, z in vertices:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for u, v in vt:
            f.write(f"vt {u:.8f} {v:.8f}\n")
        f.write("usemtl mat0\n")
        for face, uv_face in zip(faces, ft):
            f.write(
                f"f {face[0] + 1}/{uv_face[0] + 1} "
                f"{face[1] + 1}/{uv_face[1] + 1} "
                f"{face[2] + 1}/{uv_face[2] + 1}\n"
            )

    with mtl_path.open("w", encoding="ascii") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Tr 1.000000\n")
        f.write("illum 1\n")
        f.write("Ns 0.000000\n")
        f.write(f"map_Kd {texture_name}\n")


def main() -> None:
    argp = argparse.ArgumentParser(description="Export plain OBJ from Magic123 stage2 checkpoint")
    argp.add_argument(
        "--config",
        default=str(PROJECT_DIR / "configs" / "object_c.yaml"),
        help="Project-level object_c config yaml",
    )
    argp.add_argument(
        "--checkpoint",
        default=str(PROJECT_DIR / "outputs" / "object_c" / "stage2" / "checkpoints" / "stage2.pth"),
        help="Magic123 stage2 checkpoint",
    )
    argp.add_argument(
        "--workspace",
        default=str(PROJECT_DIR / "outputs" / "object_c" / "stage2"),
        help="Existing Magic123 stage2 workspace",
    )
    argp.add_argument(
        "--stage1-ckpt",
        default=str(PROJECT_DIR / "outputs" / "object_c" / "stage1" / "checkpoints" / "stage1.pth"),
        help="Stage1 checkpoint used to initialize DMTet training",
    )
    argp.add_argument(
        "--output",
        default=str(PROJECT_DIR / "outputs" / "object_c" / "object_c.obj"),
        help="Output OBJ path",
    )
    argp.add_argument(
        "--texture-output",
        default=str(PROJECT_DIR / "outputs" / "object_c" / "object_c_texture.png"),
        help="Output texture PNG path",
    )
    args = argp.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    texture_output_path = Path(args.texture_output).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.chdir(MAGIC123_DIR)

    project_cfg = load_project_config(config_path)
    opt = build_opt(args, project_cfg)
    model = make_model(opt)

    device = opt.device
    ckpt = torch.load(checkpoint_path, map_location=device)
    tet_scale = ckpt.get("tet_scale")
    if tet_scale is not None:
        model.dmtet.reset_tet_scale(torch.as_tensor(tet_scale, device=device))
    model.load_state_dict(ckpt["model"], strict=False)
    if opt.cuda_ray and "mean_density" in ckpt:
        model.mean_density = ckpt["mean_density"]

    model.eval()
    with torch.no_grad():
        vertices, faces = model.dmtet.get_verts_face()
        vertex_colors = sample_vertex_albedo(model, vertices)

    vertices_np = vertices.detach().cpu().numpy().astype(np.float32)
    faces_np = faces.detach().cpu().numpy().astype(np.int32)
    face_colors_np = (
        vertex_colors[faces.long()].mean(dim=1).detach().cpu().numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    texture_np, vt_np, ft_np = build_face_texture_atlas(faces_np, face_colors_np)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    texture_output_path.parent.mkdir(parents=True, exist_ok=True)

    final_mtl = output_path.with_suffix(".mtl")
    final_texture_kd = output_path.parent / "texture_kd.png"
    write_textured_obj(
        output_path,
        final_mtl,
        final_texture_kd.name,
        vertices_np,
        faces_np,
        vt_np,
        ft_np,
    )
    cv2.imwrite(str(final_texture_kd), cv2.cvtColor(texture_np, cv2.COLOR_RGB2BGR))
    shutil.copy2(final_texture_kd, texture_output_path)

    print(f"[export_magic123_mesh] OBJ: {output_path}")
    print(f"[export_magic123_mesh] MTL: {final_mtl}")
    print(f"[export_magic123_mesh] Texture: {final_texture_kd}")
    print(f"[export_magic123_mesh] Texture alias: {texture_output_path}")


if __name__ == "__main__":
    main()
