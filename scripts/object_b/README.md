# Object B — 文本到3D生成 (threestudio)

## 概述

使用 threestudio 框架，基于 Stable Diffusion 2.1 + SDS Loss，仅通过文本 Prompt 生成 3D 虚拟物体。

## 依赖安装

```bash
# 克隆 threestudio
git clone https://github.com/threestudio-project/threestudio.git
cd threestudio

# 安装依赖（推荐在独立的 conda 环境）
conda create -n threestudio python=3.10 -y
conda activate threestudio
pip install -r requirements.txt
```

## 运行步骤

### Step 1: 编写 Prompt

编辑 `configs/object_b.yaml`，修改 `text_prompt` 字段。

**Prompt 编写建议**：
- 使用 "A detailed ...", "A high-quality ..." 前缀
- 添加 "photorealistic", "360 degree" 等后缀
- 避免过于抽象的描述（如 "beautiful art"）

| 好 Prompt ✅ | 差 Prompt ❌ |
|-------------|-------------|
| "A detailed ceramic teapot with blue floral patterns, photorealistic" | "a teapot" |
| "A wooden treasure chest with gold trim, high quality 3D model" | "chest" |

### Step 2: 训练

```bash
bash scripts/object_b/train.sh
```

训练耗时约 40 分钟 (A6000)。

### Step 3: 转换格式

threestudio 输出 Mesh (.obj)，需转换为高斯面片：

```bash
python scripts/object_b/mesh_to_gaussian.py \
    --mesh outputs/object_b/object_b.obj \
    --texture outputs/object_b/object_b_texture.png \
    --output outputs/object_b.ply \
    --num_samples 80000
```

## 输出

| 文件 | 格式 | 说明 |
|------|------|------|
| `outputs/object_b/object_b.obj` | Wavefront OBJ | 带纹理的3D网格 |
| `outputs/object_b/object_b_texture.png` | PNG | 纹理贴图 |
| `outputs/object_b.ply` | PLY | 高斯面片格式 |

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 生成物体模糊 | 增加 `guidance_scale` 到 100-200 |
| 缺少纹理细节 | 在 Prompt 中添加材料/纹理描述 |
| 几何变形 | 增加迭代数到 15,000-20,000 |
| 显存不足 | 降低渲染分辨率 `[32, 32]` |
