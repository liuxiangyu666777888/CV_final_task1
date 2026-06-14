# Object A — 真实多视角重建 (COLMAP + 2DGS)

## 概述

使用手机环绕拍摄真实物体 → COLMAP SfM 提取相机位姿 → 2D Gaussian Splatting 重建。

## 数据采集规范

| 要求 | 说明 |
|------|------|
| 照片数量 | ≥30 张，建议 40-60 张 |
| 覆盖角度 | 360° 环绕，可包含俯拍/仰拍 |
| 重叠率 | 相邻照片重叠 > 60% |
| 光照 | 均匀漫反射光，避免强阴影 |
| 物体纹理 | 纹理丰富的物体更易重建（避免纯色/镜面） |
| 分辨率 | 建议 1920×1080 或更高 |

### 拍摄方法

1. 将物体放在稳定台面上，背景尽量简洁
2. 手机保持与物体约 30-50cm 距离
3. 以物体为中心，缓慢环绕拍摄一圈（~15秒一圈）
4. 可选：调整高度再拍一圈（俯视、平视、仰视各一圈）

## 运行步骤

### Step 1: 放置照片

将拍摄的照片（.jpg 或 .png）复制到 `data/object_a/images/`：

```bash
cp /path/to/your/photos/*.jpg data/object_a/images/
```

如果是视频格式，先抽帧：

```bash
python scripts/object_a/frame_extractor.py \
    --video /path/to/video.mp4 \
    --out data/object_a/images \
    --fps 2
```

### Step 2: 运行 COLMAP

```bash
bash scripts/object_a/run_colmap.sh
```

输出在 `data/object_a/sparse/0/`（cameras.txt, images.txt, points3D.txt）。

### Step 3: 训练 2DGS

```bash
python scripts/object_a/train_2dgs.py --config configs/object_a.yaml
```

### Step 4: 裁剪物体

打开 COLMAP GUI 或 3D 可视化工具查看点云，确定物体的边界框 (xmin, xmax, ymin, ymax, zmin, zmax)：

```bash
python scripts/object_a/crop_object.py \
    --input outputs/object_a/point_cloud/iteration_30000/point_cloud.ply \
    --output outputs/object_a.ply \
    --bbox -0.5,0.5,-0.3,0.7,-0.5,0.5
```

> **提示**: 调整 `--bbox` 参数直到只保留目标物体。

## 预期结果

- 训练耗时：~20 分钟 (A6000)
- 输出：`outputs/object_a.ply`（物体高斯面片）
- 质量指标：PSNR > 30dB, SSIM > 0.9

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| COLMAP 重建失败 | 增加照片数量、确保纹理丰富、提高重叠率 |
| 2DGS 训练显存不足 | 降采样：在 config 中设 `resolution: [800, 1200]` |
| 物体边界模糊 | 增加训练迭代到 45,000，或降低初始学习率 |
