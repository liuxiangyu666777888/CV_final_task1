#!/usr/bin/env python3
"""
preprocess.py — Remove background from a single object photo for Magic123 input.

Uses rembg (Background Removal via U²-Net) to produce an RGBA foreground image.

Usage:
    python scripts/object_c/preprocess.py \
        --input /path/to/photo.jpg \
        --output data/object_c/input_rgba.png \
        --size 512

The output is a 512×512 RGBA PNG with transparent background, suitable as
Magic123 input.
"""

import argparse
import numpy as np
from PIL import Image
import os
import sys


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using rembg, returns RGBA PIL Image."""
    from rembg import remove
    result = remove(image)
    return result


def center_crop_to_square(image: Image.Image) -> Image.Image:
    """Center-crop an image to square aspect ratio."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def preprocess_image(
    input_path: str,
    output_path: str,
    target_size: int = 512,
) -> None:
    """
    Full preprocessing pipeline:
      1. Load image
      2. Remove background (rembg)
      3. Center-crop to square
      4. Resize to target_size × target_size
      5. Save as RGBA PNG
    """
    print(f"[preprocess] Loading: {input_path}")
    image = Image.open(input_path).convert("RGB")
    print(f"  Original size: {image.size}")

    # Step 1: Background removal
    print(f"[preprocess] Removing background (rembg)...")
    rgba = remove_background(image)
    print(f"  After rembg: {rgba.size}, mode={rgba.mode}")

    # Step 2: Center-crop to square
    print(f"[preprocess] Center-cropping to square...")
    rgba = center_crop_to_square(rgba)

    # Step 3: Resize
    print(f"[preprocess] Resizing to {target_size}×{target_size}...")
    rgba = rgba.resize((target_size, target_size), Image.LANCZOS)

    # Step 4: Ensure alpha channel
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")

    # Step 5: Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rgba.save(output_path)
    print(f"[preprocess] Saved to: {output_path}")
    print(f"  Size: {rgba.size}, Mode: {rgba.mode}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess single-object photo for Magic123"
    )
    parser.add_argument("--input", required=True, help="Input photo path")
    parser.add_argument("--output", default="data/object_c/input_rgba.png",
                        help="Output RGBA PNG path")
    parser.add_argument("--size", type=int, default=512,
                        help="Target square size (default: 512)")
    parser.add_argument("--no-remove-bg", action="store_true",
                        help="Skip background removal (if already RGBA)")
    args = parser.parse_args()

    if args.no_remove_bg:
        # Just crop + resize
        print("[preprocess] Skipping background removal...")
        image = Image.open(args.input).convert("RGBA")
        image = center_crop_to_square(image)
        image = image.resize((args.size, args.size), Image.LANCZOS)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        image.save(args.output)
        print(f"[preprocess] Saved to: {args.output}")
    else:
        preprocess_image(args.input, args.output, args.size)


if __name__ == "__main__":
    main()
