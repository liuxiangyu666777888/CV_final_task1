#!/usr/bin/env python3
"""
mesh_to_gaussian.py — Convert Object B's threestudio output mesh to Gaussian Splatting PLY.

Usage:
    python scripts/object_b/mesh_to_gaussian.py \
        --mesh outputs/object_b/object_b.obj \
        --texture outputs/object_b/object_b_texture.png \
        --output outputs/object_b.ply \
        --num_samples 80000
"""

import argparse
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.mesh_to_gs import mesh_to_gaussian_splats
from utils.io_utils import save_gaussian_ply


def main():
    parser = argparse.ArgumentParser(
        description="Convert threestudio mesh to Gaussian Splatting PLY"
    )
    parser.add_argument("--mesh", required=True, help="Input .obj mesh file")
    parser.add_argument("--texture", default=None, help="Optional texture image")
    parser.add_argument("--output", required=True, help="Output .ply path")
    parser.add_argument("--num_samples", type=int, default=80000,
                        help="Number of gaussian splats to sample")
    parser.add_argument("--base_opacity", type=float, default=0.9,
                        help="Initial opacity value")
    parser.add_argument("--scale_factor", type=float, default=0.008,
                        help="Initial scale factor (smaller = finer)")
    args = parser.parse_args()

    print(f"[mesh_to_gaussian:obj_b] Converting mesh to gaussians...")
    print(f"  Mesh:   {args.mesh}")
    print(f"  Samples: {args.num_samples}")

    gs_data = mesh_to_gaussian_splats(
        mesh_path=args.mesh,
        num_samples=args.num_samples,
        texture_path=args.texture,
        base_opacity=args.base_opacity,
        scale_factor=args.scale_factor,
    )

    save_gaussian_ply(gs_data, args.output)
    print(f"[mesh_to_gaussian:obj_b] Done → {args.output}")


if __name__ == "__main__":
    main()
