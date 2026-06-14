#!/usr/bin/env bash
# =============================================================================
# train.sh — Text-to-3D generation via threestudio (Object B)
#
# Usage:
#   bash scripts/object_b/train.sh
#
# This script assumes threestudio is cloned at the project root:
#   git clone https://github.com/threestudio-project/threestudio.git
#
# Output:
#   threestudio/outputs/...          — training logs, checkpoints, previews
#   outputs/object_b/object_b.obj    — exported mesh (via a separate export step)
#   outputs/object_b/object_b.ply    — converted gaussian splats (via Step 2)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
THREESTUDIO_DIR="${PROJECT_DIR}/threestudio"
CONFIG_FILE="${PROJECT_DIR}/configs/object_b.yaml"

echo "============================================"
echo "[train:object_b] Text-to-3D via threestudio"
echo "============================================"

# ---- Check prerequisites ----
if [ ! -d "${THREESTUDIO_DIR}" ]; then
    echo "[ERROR] threestudio not found at ${THREESTUDIO_DIR}"
    echo "  Clone it first:"
    echo "  git clone https://github.com/threestudio-project/threestudio.git"
    exit 1
fi

# ---- Read config values via Python ----
read_yaml() {
    python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_FILE}')); print(c$1)"
}

PROMPT=$(read_yaml "['text_prompt']")
MODEL_NAME=$(read_yaml ".get('model_name', 'runwayml/stable-diffusion-v1-5')")
ITERATIONS=$(read_yaml "['optim']['iterations']")

echo "  Prompt:     ${PROMPT}"
echo "  Model:      ${MODEL_NAME}"
echo "  Iterations: ${ITERATIONS}"

# ---- Run threestudio ----
cd "${THREESTUDIO_DIR}"

# threestudio uses a launch.py entry point with YAML configs.
# Mesh export is a separate `--export` stage after training finishes.
python launch.py \
    --config "${THREESTUDIO_DIR}/configs/dreamfusion-sd.yaml" \
    --train \
    system.prompt_processor.pretrained_model_name_or_path="${MODEL_NAME}" \
    system.prompt_processor.prompt="${PROMPT}" \
    system.guidance.pretrained_model_name_or_path="${MODEL_NAME}" \
    trainer.max_steps="${ITERATIONS}"

cd "${PROJECT_DIR}"

echo ""
echo "[train:object_b] Training finished."
echo "[train:object_b] Next: export mesh with threestudio --export, then run mesh_to_gaussian.py"
