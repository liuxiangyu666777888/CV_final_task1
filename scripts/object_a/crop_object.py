#!/usr/bin/env python3
"""
crop_object.py — Crop a bounding box from a Gaussian Splatting PLY file.

Removes gaussians outside a user-defined axis-aligned bounding box, isolating
the foreground object from its surroundings.

Usage:
    python scripts/object_a/crop_object.py \
        --input outputs/object_a/point_cloud/iteration_30000/point_cloud.ply \
        --output outputs/object_a.ply \
        --bbox 0.0,1.0,-0.5,0.5,-0.5,0.5

The --bbox format is: x_min,x_max,y_min,y_max,z_min,z_max
"""

import argparse
import numpy as np
import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.io_utils import load_gaussian_ply, save_gaussian_ply


def parse_bbox(bbox_str: str):
    """Parse 'xmin,xmax,ymin,ymax,zmin,zmax' into (3,2) array."""
    vals = [float(v) for v in bbox_str.split(",")]
    if len(vals) != 6:
        raise ValueError("bbox must have 6 values: x_min,x_max,y_min,y_max,z_min,z_max")
    bbox = np.array([
        [vals[0], vals[1]],  # x range
        [vals[2], vals[3]],  # y range
        [vals[4], vals[5]],  # z range
    ])
    return bbox


def crop_gaussians(gs_data: dict, bbox: np.ndarray) -> dict:
    """
    Keep only gaussians whose xyz centres fall within bbox.

    Args:
        gs_data: dict from load_gaussian_ply
        bbox:    (3, 2) array [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

    Returns:
        filtered dict with same keys
    """
    xyz = gs_data["xyz"]
    mask = np.ones(xyz.shape[0], dtype=bool)
    for dim in range(3):
        mask &= (xyz[:, dim] >= bbox[dim, 0]) & (xyz[:, dim] <= bbox[dim, 1])

    n_before = xyz.shape[0]
    n_after = mask.sum()

    if n_after == 0:
        print("[crop_object] WARNING: No gaussians within bbox — check bounds!")
        return gs_data

    print(f"[crop_object] {n_before} → {n_after} gaussians ({100*n_after/n_before:.1f}% kept)")

    return {
        "xyz": xyz[mask],
        "features_dc": gs_data["features_dc"][mask],
        "opacity": gs_data["opacity"][mask],
        "scaling": gs_data["scaling"][mask],
        "rotation": gs_data["rotation"][mask],
    }


def main():
    parser = argparse.ArgumentParser(description="Crop gaussian splatting object")
    parser.add_argument("--input", required=True, help="Input .ply file")
    parser.add_argument("--output", required=True, help="Output .ply file")
    parser.add_argument("--bbox", required=True,
                        help="Bounding box: x_min,x_max,y_min,y_max,z_min,z_max")
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    gs_data = load_gaussian_ply(args.input)
    cropped = crop_gaussians(gs_data, bbox)
    save_gaussian_ply(cropped, args.output)

    print(f"[crop_object] Saved cropped gaussians to {args.output}")


if __name__ == "__main__":
    main()
