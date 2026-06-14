#!/usr/bin/env python3
"""
frame_extractor.py — Extract frames from a video for COLMAP input.

Usage:
    python frame_extractor.py --video data/object_a/video.mp4 --out data/object_a/images --fps 2

If photos are already taken as individual images, place them directly in
data/object_a/images/ and skip this script.
"""

import argparse
import cv2
import os
import sys


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> int:
    """
    Extract frames from video at given frame rate.

    Args:
        video_path: path to input video file
        output_dir: directory to save extracted frames
        fps:        frames per second to extract (default 2 = every 0.5s)

    Returns:
        number of extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Extract every Nth frame
    interval = max(1, int(video_fps / fps))

    print(f"[frame_extractor] Video: {video_path}")
    print(f"  FPS: {video_fps:.2f}, Total frames: {total_frames}, Duration: {duration:.1f}s")
    print(f"  Extracting every {interval} frame(s) → ~{fps} fps")

    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f"{count:05d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

        frame_idx += 1

    cap.release()
    print(f"[frame_extractor] Saved {count} frames to {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for COLMAP")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="data/object_a/images", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Target FPS for extraction (default: 2)")
    args = parser.parse_args()
    extract_frames(args.video, args.out, args.fps)


if __name__ == "__main__":
    main()
