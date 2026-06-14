#!/usr/bin/env python3
"""
train_2dgs.py — Train 2D Gaussian Splatting on Object A (COLMAP output).

This script wraps the 2DGS training entry point. It assumes the 2DGS repo
(https://github.com/hbb1/2d-gaussian-splatting) is cloned at the project root
or installed as a package.

Usage:
    python scripts/object_a/train_2dgs.py --config configs/object_a.yaml

The 2DGS training pipeline:
  1. Read COLMAP cameras/images/points from data/object_a/sparse/0/
  2. Initialize 2D gaussians from sparse point cloud
  3. Optimize via SDS + photometric loss for 30k iterations
  4. Save trained .ply to outputs/object_a/point_cloud/iteration_30000/
"""

import argparse
import subprocess
import os
import sys
import yaml


def find_2dgs_root() -> str:
    """Locate the 2DGS repository root."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "2d-gaussian-splatting"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "2d-gaussian-splatting"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "train.py")):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "Cannot find 2DGS repository. "
        "Clone it: git clone https://github.com/hbb1/2d-gaussian-splatting.git"
    )


def main():
    parser = argparse.ArgumentParser(description="Train 2DGS on Object A")
    parser.add_argument("--config", default="configs/object_a.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", default=None,
                        help="Override data source path (default: from config)")
    parser.add_argument("--output_dir", default=None,
                        help="Override output model path")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_source = args.data_dir or config["data"]["source_path"]
    model_path = args.output_dir or config["data"]["model_path"]
    iterations = config["optim"]["iterations"]

    os.makedirs(model_path, exist_ok=True)

    # Locate 2DGS train.py
    gs_root = find_2dgs_root()
    train_script = os.path.join(gs_root, "train.py")
    sys.path.insert(0, gs_root)

    # Build command
    cmd = [
        sys.executable, train_script,
        "-s", os.path.abspath(data_source),
        "-m", os.path.abspath(model_path),
        "--iterations", str(iterations),
        "--position_lr_init", str(config["optim"]["position_lr_init"]),
        "--position_lr_final", str(config["optim"]["position_lr_final"]),
        "--feature_lr", str(config["optim"]["feature_lr"]),
        "--opacity_lr", str(config["optim"]["opacity_lr"]),
        "--scaling_lr", str(config["optim"]["scaling_lr"]),
        "--rotation_lr", str(config["optim"]["rotation_lr"]),
        "--densification_interval", str(config["optim"]["densification_interval"]),
        "--densify_from_iter", str(config["optim"]["densify_from_iter"]),
        "--densify_until_iter", str(config["optim"]["densify_until_iter"]),
        "--lambda_dssim", str(config["optim"]["lambda_dssim"]),
    ]

    print(f"[train_2dgs] Launching 2DGS training...")
    print(f"  Data:     {data_source}")
    print(f"  Output:   {model_path}")
    print(f"  Command:  {' '.join(cmd)}")
    print("-" * 60)

    subprocess.run(cmd, check=True)

    print(f"\n[train_2dgs] Training complete!")
    print(f"  Model saved to: {model_path}/point_cloud/iteration_{iterations}/point_cloud.ply")


if __name__ == "__main__":
    main()
