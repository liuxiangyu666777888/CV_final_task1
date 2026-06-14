#!/usr/bin/env bash
# =============================================================================
# train_magic123.sh — Single-image to 3D via Magic123 (Object C)
#
# Usage:
#   bash scripts/object_c/train_magic123.sh
#
# Prerequisites:
#   git clone https://github.com/guochengqian/Magic123.git
#
# Output:
#   outputs/object_c/object_c.obj           — refined mesh (stage 2)
#   outputs/object_c/object_c_texture.png   — texture map
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MAGIC123_DIR="${PROJECT_DIR}/Magic123"
OUTPUT_DIR="${PROJECT_DIR}/outputs/object_c"
CONFIG_FILE="${PROJECT_DIR}/configs/object_c.yaml"

echo "============================================"
echo "[train:object_c] Single-Image-to-3D via Magic123"
echo "============================================"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "  HF endpoint: ${HF_ENDPOINT}"

# ---- Check prerequisites ----
if [ ! -d "${MAGIC123_DIR}" ]; then
    echo "[ERROR] Magic123 not found at ${MAGIC123_DIR}"
    echo "  Clone it first:"
    echo "  git clone https://github.com/guochengqian/Magic123.git"
    exit 1
fi

read_yaml() {
    python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_FILE}')); print(c$1)"
}

INPUT_IMAGE="${PROJECT_DIR}/$(read_yaml "['input_image']")"
TEXT_PROMPT=$(read_yaml ".get('text_prompt', '')")
STAGE1_ITERS=$(read_yaml "['stage1']['iterations']")
STAGE1_LR=$(read_yaml "['stage1']['lr']")
STAGE1_GUIDANCE_SCALE=$(read_yaml "['stage1']['guidance']['guidance_scale']")
STAGE2_ITERS=$(read_yaml "['stage2']['iterations']")
STAGE2_LR=$(read_yaml "['stage2']['lr']")
STAGE2_TET_RES=$(read_yaml "['stage2']['tet_resolution']")

if [ -z "${TEXT_PROMPT}" ] || [ "${TEXT_PROMPT}" = "None" ]; then
    BASE_NAME="$(basename "${INPUT_IMAGE}")"
    BASE_NAME="${BASE_NAME%.*}"
    BASE_NAME="${BASE_NAME%_output_rgba}"
    BASE_NAME="${BASE_NAME%_rgba}"
    BASE_NAME="${BASE_NAME//_/ }"
    TEXT_PROMPT="A high-resolution DSLR image of ${BASE_NAME}"
fi

if [ ! -f "${INPUT_IMAGE}" ]; then
    echo "[ERROR] Input image not found: ${INPUT_IMAGE}"
    echo "  Run preprocess.py first:"
    echo "  python scripts/object_c/preprocess.py --input data/object_c/cherry.jpg"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${MAGIC123_DIR}"

STAGE1_WORKSPACE="${OUTPUT_DIR}/stage1"
STAGE2_WORKSPACE="${OUTPUT_DIR}/stage2"
STAGE1_RUNNAME="$(basename "${STAGE1_WORKSPACE}")"
STAGE1_CKPT="${STAGE1_WORKSPACE}/checkpoints/${STAGE1_RUNNAME}.pth"

# ---- Stage 1: Coarse NeRF (5,000 iters) ----
if [ -f "${STAGE1_CKPT}" ]; then
    echo ""
    echo "[Stage 1/2] Existing coarse checkpoint found, skipping:"
    echo "  ${STAGE1_CKPT}"
else
    echo ""
    echo "[Stage 1/2] Coarse NeRF reconstruction (${STAGE1_ITERS} iterations)..."
    python main.py -O \
        --text "${TEXT_PROMPT}" \
        --sd_version 1.5 \
        --image "${INPUT_IMAGE}" \
        --workspace "${STAGE1_WORKSPACE}" \
        --optim adam \
        --iters "${STAGE1_ITERS}" \
        --lr "${STAGE1_LR}" \
        --guidance SD \
        --lambda_guidance 1 \
        --guidance_scale "${STAGE1_GUIDANCE_SCALE}" \
        --latent_iter_ratio 0 \
        --normal_iter_ratio 0.2 \
        --t_range 0.2 0.6 \
        --bg_radius -1
fi

if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "[ERROR] Stage 1 checkpoint not found: ${STAGE1_CKPT}"
    exit 1
fi

# ---- Stage 2: DMTet Refinement (3,000 iters) ----
echo ""
echo "[Stage 2/2] DMTet refinement (${STAGE2_ITERS} iterations)..."
set +e
python main.py -O \
    --text "${TEXT_PROMPT}" \
    --sd_version 1.5 \
    --image "${INPUT_IMAGE}" \
    --workspace "${STAGE2_WORKSPACE}" \
    --dmtet \
    --init_ckpt "${STAGE1_CKPT}" \
    --tet_grid_size "${STAGE2_TET_RES}" \
    --iters "${STAGE2_ITERS}" \
    --lr "${STAGE2_LR}" \
    --optim adam \
    --known_view_interval 4 \
    --latent_iter_ratio 0 \
    --guidance SD \
    --lambda_guidance 1e-3 \
    --guidance_scale "${STAGE1_GUIDANCE_SCALE}" \
    --rm_edge \
    --bg_radius -1
stage2_status=$?
set -e

STAGE2_RUNNAME="$(basename "${STAGE2_WORKSPACE}")"
STAGE2_CKPT="${STAGE2_WORKSPACE}/checkpoints/${STAGE2_RUNNAME}.pth"
if [ "${stage2_status}" -ne 0 ]; then
    if [ -f "${STAGE2_CKPT}" ]; then
        echo "[WARN] Stage 2 exited non-zero, but checkpoint exists:"
        echo "  ${STAGE2_CKPT}"
        echo "  Continuing with safe mesh export."
    else
        echo "[ERROR] Stage 2 failed and no checkpoint was found."
        exit "${stage2_status}"
    fi
fi

cd "${PROJECT_DIR}"

# ---- Export mesh + safe texture atlas ----
python scripts/object_c/export_magic123_mesh.py
echo "[train:object_c] Mesh exported to ${OUTPUT_DIR}/object_c.obj"
echo "[train:object_c] Texture exported to ${OUTPUT_DIR}/texture_kd.png"
echo "[train:object_c] Texture alias: ${OUTPUT_DIR}/object_c_texture.png"

echo ""
echo "============================================"
echo "[train:object_c] Done!"
echo "  Mesh: ${OUTPUT_DIR}/object_c.obj"
echo "  Texture: ${OUTPUT_DIR}/texture_kd.png"
echo "  Next: python scripts/object_c/mesh_to_gaussian.py"
echo "============================================"
