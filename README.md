# CV Final Task 1: Multi-Source 3D Asset Generation and Real-Scene Fusion

This repository contains the code for a computer vision course project that builds a complete 3D asset generation and real-scene fusion pipeline. The system reconstructs a real foreground object with COLMAP and 2D Gaussian Splatting (2DGS), generates one text-conditioned 3D asset with threestudio, generates one single-image-conditioned 3D asset with Magic123, reconstructs a real background scene with 2DGS, converts heterogeneous assets into Gaussian splats, and renders a fused roaming video.

Large datasets, model checkpoints, trained Gaussian point clouds, videos, and intermediate outputs are intentionally excluded from this repository. The expected directory layout and commands below reproduce the pipeline when the required input data and pretrained model files are prepared locally.

## Repository Structure

```text
.
├── configs/                    # Project-level YAML configs for training and fusion
├── scripts/
│   ├── object_a/               # Real multi-view object preprocessing, COLMAP, 2DGS
│   ├── object_b/               # Text-to-3D generation and mesh-to-Gaussian conversion
│   ├── object_c/               # Single-image preprocessing, Magic123, mesh export/conversion
│   ├── background/             # Mip-NeRF 360 background data and 2DGS training
│   └── fusion/                 # Gaussian merging and video rendering
├── utils/                      # Shared PLY, COLMAP, camera, and mesh-to-Gaussian utilities
├── 2d-gaussian-splatting/      # Vendored 2DGS code used by the project scripts
├── threestudio/                # Vendored threestudio code used for Object B
├── Magic123/                   # Vendored Magic123 code used for Object C
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip fallback requirements
└── README.md
```

## Requirements

The project was developed on Linux with an NVIDIA A6000 GPU. A CUDA-capable GPU is strongly recommended for 2DGS, threestudio, and Magic123.

Core system requirements:

- Linux
- NVIDIA GPU with CUDA support
- CUDA 11.7 or 11.8 compatible PyTorch
- Python 3.10
- Conda or Mamba
- COLMAP
- FFmpeg
- Git

Install system packages on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git ffmpeg colmap build-essential ninja-build
```

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate cv-final-task1
```

If PyTorch does not match your CUDA driver, reinstall it explicitly. For CUDA 11.8:

```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install local CUDA extensions and editable packages:

```bash
# 2DGS CUDA extensions
pip install -e 2d-gaussian-splatting/submodules/diff-surfel-rasterization
pip install -e 2d-gaussian-splatting/submodules/simple-knn

# Project utility package path
export PYTHONPATH="$PWD:$PWD/2d-gaussian-splatting:$PWD/threestudio:$PWD/Magic123:$PYTHONPATH"

# Magic123 custom extensions
cd Magic123
bash scripts/install_ext.sh
cd ..

# threestudio
pip install -e threestudio
```

Pip-only fallback:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation

Create the data and output folders:

```bash
mkdir -p data/object_a/images data/object_a/sparse data/object_c data/background outputs
```

### Object A: Real Multi-View Capture

Place extracted object images under:

```text
data/object_a/images/
```

If the input is a video, extract frames first:

```bash
python scripts/object_a/frame_extractor.py \
  --video /path/to/object_a_video.mp4 \
  --out data/object_a/images \
  --fps 2
```

Then run COLMAP:

```bash
bash scripts/object_a/run_colmap.sh
```

The expected COLMAP output is:

```text
data/object_a/sparse/0/
```

### Object B: Text Prompt

Edit the text prompt and generation settings in:

```text
configs/object_b.yaml
```

The default prompt is:

```text
A wooden treasure chest with gold trim, photorealistic, 360 degree
```

The first run downloads diffusion model weights through Hugging Face. Configure your Hugging Face cache or mirror if needed.

### Object C: Single Image

Place the input image under `data/object_c/`, or pass an absolute path to the preprocessing command. The default config expects:

```text
data/object_c/cherry_output_rgba.png
```

Generate an RGBA foreground image:

```bash
python scripts/object_c/preprocess.py \
  --input /path/to/object_c_image.jpg \
  --output data/object_c/cherry_output_rgba.png
```

Magic123 requires Zero123 and Stable Diffusion weights. Follow the upstream Magic123 instructions for downloading required pretrained weights into the expected local cache/pretrained locations. Model weights are not included in this repository.

### Background Scene

The pipeline uses the Mip-NeRF 360 `garden` scene as the real background. Download it with:

```bash
bash scripts/background/download_scene.sh garden
```

The expected scene directory is:

```text
data/background/garden/
```

## Train Commands

Run commands from the repository root after activating the environment.

### Train Object A with 2DGS

```bash
python scripts/object_a/train_2dgs.py --config configs/object_a.yaml
```

Optional held-out evaluation run using the 2DGS evaluation split:

```bash
python 2d-gaussian-splatting/train.py \
  -s data/object_a \
  -m outputs/object_a_eval \
  --eval \
  --iterations 30000 \
  --position_lr_init 0.00016 \
  --position_lr_final 0.0000016 \
  --feature_lr 0.0025 \
  --opacity_lr 0.05 \
  --scaling_lr 0.005 \
  --rotation_lr 0.001 \
  --densification_interval 100 \
  --densify_from_iter 500 \
  --densify_until_iter 15000 \
  --lambda_dssim 0.2 \
  --test_iterations 1000 3000 7000 15000 30000 \
  --save_iterations 7000 30000
```

Crop the reconstructed object:

```bash
python scripts/object_a/crop_object.py \
  --input outputs/object_a/point_cloud/iteration_30000/point_cloud.ply \
  --output outputs/object_a_tight.ply \
  --bbox=-1.95,-0.45,0.15,1.85,2.05,3.35
```

### Train Object B with threestudio

```bash
export HF_ENDPOINT=https://hf-mirror.com  # optional, use only when needed
bash scripts/object_b/train.sh
```

Convert the exported textured mesh to Gaussian splats:

```bash
python scripts/object_b/mesh_to_gaussian.py \
  --mesh outputs/object_b/object_b.obj \
  --texture outputs/object_b/object_b_texture.png \
  --output outputs/object_b.ply
```

### Train Object C with Magic123

```bash
bash scripts/object_c/train_magic123.sh
```

Export the Magic123 mesh if needed:

```bash
python scripts/object_c/export_magic123_mesh.py \
  --workspace outputs/object_c/stage2 \
  --output outputs/object_c/object_c.obj \
  --texture-output outputs/object_c/object_c_texture.png
```

Convert the mesh to Gaussian splats:

```bash
python scripts/object_c/mesh_to_gaussian.py \
  --mesh outputs/object_c/object_c.obj \
  --texture outputs/object_c/object_c_texture.png \
  --output outputs/object_c.ply
```

### Train Background with 2DGS

```bash
python scripts/background/train_2dgs.py --config configs/background.yaml
```

## Test and Rendering Commands

### Render 2DGS Test Views

For a held-out Object A evaluation run:

```bash
python 2d-gaussian-splatting/render.py \
  -m outputs/object_a_eval \
  --iteration 30000 \
  --skip_train \
  --skip_mesh
```

The rendered held-out views are written to:

```text
outputs/object_a_eval/test/ours_30000/
```

### Compute 2DGS Metrics

```bash
python 2d-gaussian-splatting/metrics.py \
  -m outputs/object_a_eval
```

### Merge Foreground Objects with Background

Before merging, make sure the following files exist:

```text
outputs/object_a_tight.ply
outputs/object_b.ply
outputs/object_c.ply
outputs/background/point_cloud/iteration_30000/point_cloud.ply
```

Then run:

```bash
python scripts/fusion/merge_gaussians.py \
  --config configs/fusion.yaml \
  --output outputs/merged.ply
```

### Render the Final Roaming Video

```bash
python scripts/fusion/render_video.py --config configs/fusion.yaml
```

The final video is written to:

```text
outputs/roaming.mp4
```

## Reproducibility Notes

- `configs/object_a.yaml`, `configs/background.yaml`, and the explicit `--eval` command above use 30,000 2DGS optimization iterations.
- `configs/object_b.yaml` uses a Stable Diffusion guidance setup for SDS-based text-to-3D generation.
- `configs/object_c.yaml` follows a two-stage Magic123 workflow: coarse NeRF optimization and DMTet refinement.
- `configs/fusion.yaml` stores the final object scales, rotations, translations, background filtering thresholds, and rendering camera path.

## Files Intentionally Not Tracked

The following categories are excluded by `.gitignore`:

- Raw datasets under `data/`
- Training outputs under `outputs/`
- Gaussian splats, meshes, videos, checkpoints, tensorboard logs, and pretrained weights
- Hugging Face/model caches
- Python caches and compiled CUDA extension build products

This keeps the repository code-focused and avoids committing large files or licensed datasets/model weights.

## Third-Party Code

This repository vendors code from:

- 2D Gaussian Splatting
- threestudio
- Magic123

Their original licenses and READMEs are preserved in the corresponding subdirectories. Use the third-party components according to their upstream licenses.
