#!/usr/bin/env python3
"""
train_2dgs.py — Train 2DGS on Mip-NeRF 360 background scene.

Similar to Object A training, but for the larger background scene.
Mip-NeRF 360 scenes come with COLMAP poses precomputed, so no SfM needed.

Usage:
    python scripts/background/train_2dgs.py --config configs/background.yaml
"""

import argparse
import subprocess
import os
import sys
import yaml


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_2dgs_root() -> str:
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "2d-gaussian-splatting"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "2d-gaussian-splatting"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "train.py")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Cannot find 2DGS repository. "
        "Clone: git clone https://github.com/hbb1/2d-gaussian-splatting.git"
    )


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


def validate_scene_root(scene_root: str) -> None:
    sparse_root = os.path.join(scene_root, "sparse", "0")
    required = [
        os.path.join(sparse_root, "cameras.bin"),
        os.path.join(sparse_root, "images.bin"),
        os.path.join(sparse_root, "points3D.bin"),
    ]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        missing_str = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Background scene is incomplete for 2DGS.\n"
            "Expected COLMAP files under data/background/<scene>/sparse/0:\n"
            f"{missing_str}\n"
            "Re-run:\n"
            "  bash scripts/background/download_scene.sh garden"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train 2DGS on Mip-NeRF 360 background scene"
    )
    parser.add_argument("--config", default="configs/background.yaml")
    parser.add_argument("--scene", default="garden",
                        help="Scene name (garden, bicycle, counter, ...)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_source = resolve_project_path(config["data"]["source_path"])
    model_path = resolve_project_path(config["data"]["model_path"])
    iterations = config["optim"]["iterations"]
    scene_root = data_source

    os.makedirs(model_path, exist_ok=True)
    validate_scene_root(scene_root)

    gs_root = find_2dgs_root()
    train_script = os.path.join(gs_root, "train.py")

    cmd = [
        sys.executable, train_script,
        "-s", scene_root,
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

    print(f"[train_2dgs:background] Training on {args.scene}")
    print(f"  Data:   {data_source}")
    print(f"  Output: {model_path}")
    subprocess.run(cmd, check=True, cwd=gs_root)

    print(f"\n[train_2dgs:background] Done!")
    print(f"  PLY: {model_path}/point_cloud/iteration_{iterations}/point_cloud.ply")


if __name__ == "__main__":
    main()
