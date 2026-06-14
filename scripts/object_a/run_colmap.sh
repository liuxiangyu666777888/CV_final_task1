#!/usr/bin/env bash
# =============================================================================
# run_colmap.sh — Run COLMAP SfM pipeline for Object A
#
# Usage:
#   bash scripts/object_a/run_colmap.sh
#
# Prerequisites:
#   - COLMAP installed and available on PATH
#   - Images placed in data/object_a/images/
#
# Output:
#   data/object_a/sparse/0/  — cameras.bin, images.bin, points3D.bin
#   data/object_a/database.db
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_DIR="${PROJECT_DIR}/data/object_a/images"
SPARSE_DIR="${PROJECT_DIR}/data/object_a/sparse"
DB_PATH="${PROJECT_DIR}/data/object_a/database.db"

echo "============================================"
echo "[run_colmap] Starting COLMAP SfM pipeline"
echo "  Images:  ${IMAGE_DIR}"
echo "  Sparse:  ${SPARSE_DIR}"
echo "  DB:      ${DB_PATH}"
echo "============================================"

# Check prerequisites
if ! command -v colmap &> /dev/null; then
    echo "[ERROR] colmap not found. Install: https://colmap.github.io/"
    exit 1
fi

if [ ! -d "${IMAGE_DIR}" ] || [ -z "$(ls -A "${IMAGE_DIR}")" ]; then
    echo "[ERROR] No images in ${IMAGE_DIR}"
    exit 1
fi

NUM_IMAGES=$(find "${IMAGE_DIR}" -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
echo "[run_colmap] Found ${NUM_IMAGES} images"

# Set offscreen rendering for headless servers (no display)
export QT_QPA_PLATFORM=offscreen

mkdir -p "${SPARSE_DIR}"

# -------------------------------------------------------------------
# Step 1: Feature extraction
# -------------------------------------------------------------------
echo ""
echo "[Step 1/4] Feature extraction..."
colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_DIR}" \
    --ImageReader.camera_model SIMPLE_PINHOLE \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0

# -------------------------------------------------------------------
# Step 2: Exhaustive matching
# -------------------------------------------------------------------
echo ""
echo "[Step 2/4] Exhaustive feature matching..."
colmap exhaustive_matcher \
    --database_path "${DB_PATH}" \
    --SiftMatching.use_gpu 0

# -------------------------------------------------------------------
# Step 3: Sparse reconstruction (mapper)
# -------------------------------------------------------------------
echo ""
echo "[Step 3/4] Sparse reconstruction..."
colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGE_DIR}" \
    --output_path "${SPARSE_DIR}"

# -------------------------------------------------------------------
# Step 4: Convert binary → text (for 2DGS compatibility)
# -------------------------------------------------------------------
echo ""
echo "[Step 4/4] Convert model to text format..."
MODEL_DIR="${SPARSE_DIR}/0"
if [ -d "${MODEL_DIR}" ]; then
    colmap model_converter \
        --input_path "${MODEL_DIR}" \
        --output_path "${MODEL_DIR}" \
        --output_type TXT
    echo "[run_colmap] Text-format model written to ${MODEL_DIR}"
else
    echo "[ERROR] Sparse model not found at ${MODEL_DIR}"
    echo "  Check if COLMAP reconstruction succeeded."
    exit 1
fi

echo ""
echo "============================================"
echo "[run_colmap] Done! SfM model at ${MODEL_DIR}"
echo "  cameras.txt, images.txt, points3D.txt"
echo "============================================"
