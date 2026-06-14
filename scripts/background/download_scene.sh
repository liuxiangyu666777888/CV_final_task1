#!/usr/bin/env bash
# =============================================================================
# download_scene.sh — Download a Mip-NeRF 360 scene
#
# Usage:
#   bash scripts/background/download_scene.sh [scene_name]
#
# Default scene: garden
# Available: garden, bicycle, counter, bonsai, kitchen, room, stump, treehill
#
# Output: data/background/<scene_name>/
# =============================================================================

set -euo pipefail

SCENE="${1:-garden}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/background"

echo "============================================"
echo "[download_scene] Mip-NeRF 360 — ${SCENE}"
echo "============================================"

mkdir -p "${DATA_DIR}"

# Mip-NeRF 360 dataset URL (Google Drive or official mirror)
# Using the official multi-scale download link:
BASE_URL="https://storage.googleapis.com/gresearch/refraw360"

# For garden scene
ZIP_FILE="${SCENE}.zip"
DOWNLOAD_URL="${BASE_URL}/${ZIP_FILE}"

echo "  Downloading: ${DOWNLOAD_URL}"

# Download with wget or curl
if command -v wget &> /dev/null; then
    wget -c "${DOWNLOAD_URL}" -O "${DATA_DIR}/${ZIP_FILE}"
elif command -v curl &> /dev/null; then
    curl -C - -L "${DOWNLOAD_URL}" -o "${DATA_DIR}/${ZIP_FILE}"
else
    echo "[ERROR] Neither wget nor curl found."
    exit 1
fi

# Extract
echo "  Extracting to ${DATA_DIR}/${SCENE}..."
mkdir -p "${DATA_DIR}/${SCENE}"
unzip -o "${DATA_DIR}/${ZIP_FILE}" -d "${DATA_DIR}/${SCENE}"

# Clean up zip
rm "${DATA_DIR}/${ZIP_FILE}"

echo "  Done! Scene at: ${DATA_DIR}/${SCENE}"
echo ""
echo "  Next: Prepare 2DGS format and train:"
echo "  python scripts/background/train_2dgs.py"
