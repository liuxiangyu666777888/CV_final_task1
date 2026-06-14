# Object C — 单图到3D生成 (Magic123)

## 概述

拍摄一张真实物体的单张照片 → 去背景 → Magic123 两阶段生成 → 转换为高斯面片。

## 拍摄规范

| 要求 | 说明 |
|------|------|
| 角度 | 正面拍摄，物体居中 |
| 光照 | 均匀，避免强烈反光 |
| 纹理 | 纹理丰富的物体效果更好 |
| 背景 | 简单的纯色背景有助于自动去背景 |

## 运行步骤

### Step 1: 预处理（去背景）

```bash
python scripts/object_c/preprocess.py \
    --input /path/to/your/photo.jpg \
    --output data/object_c/input_rgba.png \
    --size 512
```

### Step 2: Magic123 训练

```bash
bash scripts/object_c/train_magic123.sh
```

- Stage 1: 粗 NeRF 重建 (5,000 次迭代, ~25 分钟)
- Stage 2: DMTet 精修 (3,000 次迭代, ~25 分钟)
- 总耗时: ~50 分钟 (A6000)

### Step 3: 转换格式

```bash
python scripts/object_c/mesh_to_gaussian.py \
    --mesh outputs/object_c/object_c.obj \
    --texture outputs/object_c/object_c_texture.png \
    --output outputs/object_c.ply \
    --num_samples 80000
```

## 输出

| 文件 | 说明 |
|------|------|
| `data/object_c/input_rgba.png` | 预处理后的RGBA前景图 |
| `outputs/object_c/object_c.obj` | 精修后的3D网格 + 纹理 |
| `outputs/object_c.ply` | 高斯面片格式 |

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| rembg 去背景不干净 | 手动用 Photoshop/GIMP 处理，或使用 SAM 进行精细分割 |
| 背面纹理模糊 | Magic123 仅依赖单视角，背面属正常现象；选择不需要展示背面的物体 |
| 生成物体与输入不一致 | 增加 stage1 迭代数、调高 guidance_scale |
