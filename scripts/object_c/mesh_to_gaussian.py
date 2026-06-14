#!/usr/bin/env python3
"""
mesh_to_gaussian.py — Convert Object C's Magic123 output mesh to Gaussian Splatting PLY.

Usage:
    python scripts/object_c/mesh_to_gaussian.py \
        --mesh outputs/object_c/object_c.obj \
        --texture outputs/object_c/object_c_texture.png \
        --output outputs/object_c.ply \
        --num_samples 80000
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.mesh_to_gs import mesh_to_gaussian_splats
from utils.io_utils import save_gaussian_ply


def estimate_foreground_rgb(image_path: str) -> np.ndarray:
    rgba = np.array(Image.open(image_path).convert("RGBA"), dtype=np.float32) / 255.0
    mask = rgba[:, :, 3] > 0.125
    if not np.any(mask):
        raise ValueError(f"No foreground alpha found in {image_path}")
    return rgba[:, :, :3][mask].mean(axis=0).astype(np.float32)


def rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    c0 = 0.28209479177387814
    return ((rgb - 0.5) / c0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Magic123 mesh to Gaussian Splatting PLY"
    )
    parser.add_argument("--mesh", required=True, help="Input .obj mesh file")
    parser.add_argument("--texture", default=None, help="Optional texture image")
    parser.add_argument("--output", required=True, help="Output .ply path")
    parser.add_argument("--num_samples", type=int, default=80000,
                        help="Number of gaussian splats")
    parser.add_argument("--base_opacity", type=float, default=0.9)
    parser.add_argument("--scale_factor", type=float, default=0.008)
    parser.add_argument(
        "--color_image",
        default=None,
        help="Optional RGBA image used to force a solid foreground color",
    )
    args = parser.parse_args()

    print(f"[mesh_to_gaussian:obj_c] Converting...")
    gs_data = mesh_to_gaussian_splats(
        mesh_path=args.mesh,
        num_samples=args.num_samples,
        texture_path=args.texture,
        base_opacity=args.base_opacity,
        scale_factor=args.scale_factor,
    )

    if args.color_image and os.path.exists(args.color_image):
        mean_rgb = estimate_foreground_rgb(args.color_image)
        sh_dc = rgb_to_sh_dc(mean_rgb)
        gs_data["features_dc"][:] = sh_dc.reshape(1, 3, 1)
        print(f"[mesh_to_gaussian:obj_c] Applied solid color from {args.color_image}: {mean_rgb.tolist()}")

    save_gaussian_ply(gs_data, args.output)
    print(f"[mesh_to_gaussian:obj_c] Done → {args.output}")


if __name__ == "__main__":
    main()
