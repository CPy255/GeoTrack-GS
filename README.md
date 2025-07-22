# GeoTrack-GS

本项目利用 3D 高斯溅射（3D Gaussian Splatting）技术，结合先进的几何约束系统和GT-DCA增强外观建模，实现高质量的 3D 场景重建与渲染。该项目基于论文《3D Gaussian Splatting for Real-Time Radiance Field Rendering》，并集成了多尺度几何约束、自适应权重调整和GT-DCA外观增强等创新功能。

## 主要功能 (Features)

*   **高质量实时渲染**: 基于 3D Gaussian Splatting，实现照片级的实时渲染效果
*   **GT-DCA外观增强**: 集成GT-DCA（Geometry-guided Track-based Deformable Cross-Attention）模块，提供增强的外观建模能力
*   **几何约束系统**: 集成多尺度几何约束，提升重建精度和一致性
*   **自适应权重调整**: 动态调整约束权重，优化训练过程
*   **轨迹管理**: 智能相机轨迹管理，提升多视角一致性
*   **重投影验证**: 实时重投影误差验证，确保几何准确性
*   **端到端工作流**: 支持从 COLMAP 数据集直接进行训练、渲染和评估
*   **置信度评估**: 集成置信度信息，提升渲染的稳定性和准确性
*   **PyTorch 核心**: 完全使用 PyTorch 构建，易于理解、修改和扩展

## 先决条件 (Prerequisites)

在开始之前，请确保你的系统满足以下要求：
*   **操作系统**: Linux (推荐) 或 Windows
*   **GPU**: 支持 CUDA 11.8 或更高版本的 NVIDIA GPU (推荐 RTX 3080 或更高)
*   **内存**: 至少 16GB RAM，推荐 32GB
*   **存储**: 至少 10GB 可用空间
*   **软件**:
    *   Anaconda 或 Miniconda
    *   Git
    *   NVIDIA CUDA Toolkit 11.8+
    *   Visual Studio Build Tools (Windows)

## 安装 (Installation)

### 1. 克隆仓库

```bash
# 注意：--recursive 参数是必需的，用于克隆所有子模块
git clone --recursive https://github.com/CPy255/GeoTrack-GS.git
cd GeoTrack-GS
```

### 2. 创建并激活 Conda 环境

```bash
conda env create -f environment.yml
conda activate geotrack
```

### 3. 安装 PyTorch CUDA 环境

确保安装支持 CUDA 的 PyTorch 版本：

```bash
# 对于 CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 对于 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证 CUDA 安装
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

### 4. 构建自定义 CUDA 子模块

本项目需要自定义的 CUDA 核心。请运行以下命令来构建和安装它们：

```bash
# 构建差分高斯光栅化模块
cd submodules/diff-gaussian-rasterization-confidence
python setup.py install
cd ../..

# 构建简单 KNN 模块
cd submodules/simple-knn
python setup.py install
cd ../..
```

### 5. 验证安装

运行测试脚本验证所有模块是否正确安装：

```bash
python debug/test_modules.py
```

## 使用方法 (Usage)

### 1. 准备数据

#### 数据集要求
你需要一个由 COLMAP 处理过的数据集。目录结构通常应包含一个 `images` 文件夹和一个带有 COLMAP 重建结果的 `sparse` 文件夹。

#### 快速开始 - 下载示例数据集
为了方便快速开始，你可以下载官方提供的示例数据集：
```bash
# (可选) 下载示例数据到 data/ 目录
git clone https://github.com/graphdeco-inria/gaussian-splatting-data.git data
```
*注意：该数据集较大，下载可能需要一些时间。*

#### 数据预处理 - COLMAP 处理

如果你有自己的图像数据，需要使用 COLMAP 进行预处理。本项目支持两种主要的数据集格式：

##### LLFF 数据集处理
对于前向面向场景（如街景、建筑物正面等），使用 LLFF 格式：

```bash
# 安装 COLMAP (如果尚未安装)
# Ubuntu/Debian:
sudo apt-get install colmap

# Windows: 下载并安装 COLMAP from https://colmap.github.io/

# 使用 tools/colmap_llff.py 脚本处理 LLFF 数据集
python tools/colmap_llff.py -s /path/to/your/images

# 指定输出路径
python tools/colmap_llff.py -s /path/to/your/images -o /path/to/output

# 高质量 LLFF 处理
python tools/colmap_llff.py -s /path/to/your/images -o /path/to/output --quality high --feature_type sift
```

##### 360度数据集处理
对于 360 度环绕场景（如物体中心拍摄），使用 360 格式：

```bash
# 处理 360 度数据集
python tools/colmap_360.py -s /path/to/your/images

# 指定输出路径
python tools/colmap_360.py -s /path/to/your/images -o /path/to/output

# 对于高质量 360 度重建
python tools/colmap_360.py -s /path/to/your/images \
    -o /path/to/output \
    --quality high \
    --feature_type sift \
    --matcher_type exhaustive
```

##### 手动 COLMAP 处理
如果需要更精细的控制，可以手动运行 COLMAP：

```bash
# 1. 特征提取
colmap feature_extractor \
    --database_path /path/to/database.db \
    --image_path /path/to/images \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.use_gpu 1

# 2. 特征匹配
colmap exhaustive_matcher \
    --database_path /path/to/database.db \
    --SiftMatching.use_gpu 1

# 3. 稀疏重建
colmap mapper \
    --database_path /path/to/database.db \
    --image_path /path/to/images \
    --output_path /path/to/sparse

# 4. 图像去畸变（可选，用于密集重建）
colmap image_undistorter \
    --image_path /path/to/images \
    --input_path /path/to/sparse/0 \
    --output_path /path/to/dense \
    --output_type COLMAP
```

##### 数据集验证
处理完成后，验证数据集结构：

```bash
# 检查数据集结构
python -c "
import os
dataset_path = '/path/to/your/dataset'
required_files = ['images', 'sparse/0/cameras.bin', 'sparse/0/images.bin', 'sparse/0/points3D.bin']
for file in required_files:
    path = os.path.join(dataset_path, file)
    if os.path.exists(path):
        print(f'✓ {file} 存在')
    else:
        print(f'✗ {file} 缺失')
"

# 查看相机参数
python -c "
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
cameras = read_intrinsics_binary('/path/to/your/dataset/sparse/0/cameras.bin')
images = read_extrinsics_binary('/path/to/your/dataset/sparse/0/images.bin')
print(f'相机数量: {len(cameras)}')
print(f'图像数量: {len(images)}')
"
```

##### 数据集质量优化建议

**拍摄建议：**
- **LLFF 场景**: 保持相机朝向一致，适度的视差变化
- **360 场景**: 围绕物体均匀拍摄，保持距离一致
- **重叠度**: 相邻图像至少 60-80% 重叠
- **光照**: 保持一致的光照条件
- **分辨率**: 推荐 1080p 或更高分辨率

**COLMAP 参数优化：**

```bash
# LLFF 高质量处理
python tools/colmap_llff.py -s /path/to/images \
    -o /path/to/output \
    --quality high \
    --feature_type sift \
    --num_threads 8

# 360 度高质量处理
python tools/colmap_360.py -s /path/to/images \
    -o /path/to/output \
    --quality high \
    --feature_type sift \
    --matcher_type exhaustive

# 针对困难场景的参数
python tools/colmap_llff.py -s /path/to/images \
    -o /path/to/output \
    --quality extreme \
    --feature_type sift \
    --matcher_type exhaustive \
    --ba_refine_focal_length \
    --ba_refine_principal_point
```

##### 故障排除

**常见问题及解决方案：**

```bash
# 如果 COLMAP 重建失败，尝试降低质量要求
python tools/colmap_llff.py -s /path/to/images --quality medium --feature_type orb

# 如果图像过多导致内存不足
python tools/colmap_360.py -s /path/to/images --max_num_images 200 --quality medium

# 如果特征匹配失败，使用顺序匹配
python tools/colmap_llff.py -s /path/to/images --matcher_type sequential --overlap 10

# 检查处理日志
python tools/colmap_360.py -s /path/to/images --verbose

# 处理完成后验证数据集
python tools/colmap_llff.py -s /path/to/processed/dataset --validate_only
```

##### 处理示例

**完整的数据处理流程：**

```bash
# 1. LLFF 数据集处理示例
python tools/colmap_llff.py -s /path/to/llff/images -o /path/to/llff/output

# 2. 360 度数据集处理示例
python tools/colmap_360.py -s /path/to/360/images -o /path/to/360/output

# 3. 批量处理多个数据集
for dataset in dataset1 dataset2 dataset3; do
    python tools/colmap_llff.py -s /path/to/$dataset/images -o /path/to/$dataset/processed
done

# 4. 处理后直接训练
python tools/colmap_llff.py -s /path/to/images -o /path/to/processed
python train.py -s /path/to/processed -m output/model
```

### 2. 训练

#### 基础训练
使用 `train.py` 脚本来训练一个新模型：

```bash
# 基础训练示例
python train.py -s data/tandt/train -m output/tandt

# 高质量训练（更多迭代）
python train.py -s data/tandt/train -m output/tandt_hq --iterations 30000

# 使用几何约束的训练
python train.py -s data/tandt/train -m output/tandt_geo --enable_geometric_constraints --constraint_weight 0.1
```

#### GT-DCA增强外观建模训练
启用GT-DCA模块进行增强的外观建模：

```bash
# 启用GT-DCA增强外观建模
python train.py -s data/tandt/train -m output/tandt_gtdca \
    --use_gt_dca \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8

# GT-DCA高质量训练
python train.py -s data/tandt/train -m output/tandt_gtdca_hq \
    --use_gt_dca \
    --gt_dca_feature_dim 512 \
    --gt_dca_num_sample_points 16 \
    --gt_dca_attention_heads 8 \
    --gt_dca_enable_caching

# 结合几何约束和GT-DCA的完整训练
python train.py -s data/tandt/train -m output/tandt_full \
    --enable_geometric_constraints \
    --use_gt_dca \
    --multiscale_constraints \
    --adaptive_weighting \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8 \
    --constraint_weight 0.1 \
    --iterations 25000
```

#### 几何约束训练
启用几何约束系统进行更精确的重建：

```bash
# 启用多尺度几何约束
python train.py -s data/tandt/train -m output/tandt_multiscale \
    --enable_geometric_constraints \
    --multiscale_constraints \
    --constraint_weight 0.15 \
    --adaptive_weighting

# 启用轨迹管理和重投影验证
python train.py -s data/tandt/train -m output/tandt_trajectory \
    --enable_geometric_constraints \
    --trajectory_management \
    --reprojection_validation \
    --constraint_weight 0.2

# 完整几何约束训练（推荐用于高质量重建）
python train.py -s data/tandt/train -m output/tandt_full_geo \
    --enable_geometric_constraints \
    --multiscale_constraints \
    --adaptive_weighting \
    --trajectory_management \
    --reprojection_validation \
    --constraint_weight 0.1 \
    --iterations 25000
```

#### 训练参数说明

**基础参数:**
*   `-s, --source_path`: 输入数据集所在的目录路径
*   `-m, --model_path`: 用于保存训练好的模型和检查点的目录路径
*   `--iterations`: 训练迭代次数 (默认: 30000)

**GT-DCA参数:**
*   `--use_gt_dca`: 启用GT-DCA增强外观建模
*   `--gt_dca_feature_dim`: GT-DCA特征维度 (默认: 256)
*   `--gt_dca_num_sample_points`: 可变形采样点数量 (默认: 8)
*   `--gt_dca_hidden_dim`: GT-DCA隐藏层维度 (默认: 128)
*   `--gt_dca_attention_heads`: 交叉注意力头数 (默认: 8)
*   `--gt_dca_confidence_threshold`: 轨迹点置信度阈值 (默认: 0.5)
*   `--gt_dca_min_track_points`: 最小轨迹点数量 (默认: 4)
*   `--gt_dca_enable_caching`: 启用GT-DCA特征缓存
*   `--gt_dca_dropout_rate`: GT-DCA模块的Dropout率 (默认: 0.1)

**几何约束参数:**
*   `--enable_geometric_constraints`: 启用几何约束系统
*   `--multiscale_constraints`: 启用多尺度约束
*   `--adaptive_weighting`: 启用自适应权重调整
*   `--trajectory_management`: 启用轨迹管理
*   `--reprojection_validation`: 启用重投影验证
*   `--constraint_weight`: 几何约束权重 (默认: 0.1)

要查看所有可用的训练选项，请运行：
```bash
python train.py --help
```

### 3. 渲染

#### 基础渲染
当您拥有一个训练好的模型后，就可以从新的摄像机视角渲染图像：

```bash
# 基础渲染
python render.py -m output/tandt

# 高质量渲染
python render.py -m output/tandt --render_quality high

# 渲染测试集
python render.py -m output/tandt --skip_train --skip_test
```

#### GT-DCA增强渲染
使用GT-DCA模块进行增强的外观渲染：

```bash
# 启用GT-DCA增强渲染
python render.py -m output/tandt_gtdca --use_gt_dca

# GT-DCA高质量渲染
python render.py -m output/tandt_gtdca_hq \
    --use_gt_dca \
    --gt_dca_feature_dim 512 \
    --gt_dca_num_sample_points 16 \
    --gt_dca_enable_caching

# 结合几何约束和GT-DCA的完整渲染
python render.py -m output/tandt_full \
    --use_gt_dca \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8
```

#### 几何约束渲染
使用几何约束进行更稳定的渲染：

```bash
# 启用几何约束渲染
python render.py -m output/tandt_geo --enable_geometric_constraints

# 启用重投影验证的渲染
python render.py -m output/tandt_geo \
    --enable_geometric_constraints \
    --reprojection_validation \
    --validation_threshold 2.0

# 多尺度约束渲染
python render.py -m output/tandt_multiscale \
    --enable_geometric_constraints \
    --multiscale_constraints
```

#### 渲染参数说明

**基础参数:**
*   `-m, --model_path`: 训练好的模型路径
*   `--skip_train`: 跳过训练集渲染
*   `--skip_test`: 跳过测试集渲染

**GT-DCA参数:**
*   `--use_gt_dca`: 启用GT-DCA增强渲染
*   `--gt_dca_feature_dim`: GT-DCA特征维度 (默认: 256)
*   `--gt_dca_num_sample_points`: 可变形采样点数量 (默认: 8)
*   `--gt_dca_hidden_dim`: GT-DCA隐藏层维度 (默认: 128)
*   `--gt_dca_attention_heads`: 交叉注意力头数 (默认: 8)
*   `--gt_dca_enable_caching`: 启用GT-DCA特征缓存

**几何约束参数:**
*   `--enable_geometric_constraints`: 启用几何约束
*   `--reprojection_validation`: 启用重投影验证
*   `--multiscale_constraints`: 启用多尺度约束
*   `--validation_threshold`: 验证阈值 (默认: 1.0)

### 4. 评估

#### 基础评估
要评估一个训练好的模型，请使用 `full_eval.py` 脚本：

```bash
# 基础评估
python full_eval.py -m output/tandt

# 详细评估报告
python full_eval.py -m output/tandt --detailed_report
```

#### 几何约束评估
评估几何约束模型的性能：

```bash
# 几何约束评估
python full_eval.py -m output/tandt_geo \
    --enable_geometric_constraints \
    --geometric_metrics

# 多尺度约束评估
python full_eval.py -m output/tandt_multiscale \
    --enable_geometric_constraints \
    --multiscale_constraints \
    --geometric_metrics \
    --detailed_report

# 完整评估（包含所有指标）
python full_eval.py -m output/tandt_full_geo \
    --enable_geometric_constraints \
    --geometric_metrics \
    --reprojection_metrics \
    --trajectory_metrics \
    --detailed_report
```

#### GT-DCA评估
评估GT-DCA增强外观建模的效果：

```bash
# GT-DCA模型评估
python full_eval.py -m output/tandt_gtdca

# GT-DCA高质量模型评估
python full_eval.py -m output/tandt_gtdca_hq --detailed_report

# 结合几何约束和GT-DCA的完整评估
python full_eval.py -m output/tandt_full \
    --detailed_report
```

#### 评估参数说明
*   `-m, --model_path`: 要评估的模型路径
*   `--enable_geometric_constraints`: 启用几何约束评估
*   `--geometric_metrics`: 计算几何相关指标
*   `--reprojection_metrics`: 计算重投影误差指标
*   `--trajectory_metrics`: 计算轨迹一致性指标
*   `--detailed_report`: 生成详细评估报告

### 5. 配置文件

项目支持通过配置文件自定义几何约束参数：

```bash
# 使用自定义配置文件训练
python train.py -s data/tandt/train -m output/tandt_custom \
    --config config/constraints.json

# 查看默认配置
python -c "from geometric_constraints.config import load_config; print(load_config())"
```

### 6. GT-DCA详细说明

#### GT-DCA模块介绍
GT-DCA（Geometry-guided Track-based Deformable Cross-Attention）是本项目的核心创新功能，它通过以下两个阶段提供增强的外观建模：

**阶段1: 几何引导 (Geometry Guidance)**
- 从3D高斯基元生成基础外观特征作为查询向量
- 使用交叉注意力机制注入几何上下文信息
- 利用2D轨迹点提供几何引导

**阶段2: 可变形采样 (Deformable Sampling)**
- 预测采样偏移量和注意力权重
- 从2D特征图进行可变形采样
- 聚合加权特征生成最终的增强外观特征

#### GT-DCA配置建议

**基础配置（适用于大多数场景）:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 256 \
    --gt_dca_num_sample_points 8 \
    --gt_dca_attention_heads 8
```

**高质量配置（适用于复杂场景）:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 512 \
    --gt_dca_num_sample_points 16 \
    --gt_dca_attention_heads 16 \
    --gt_dca_enable_caching \
    --gt_dca_dropout_rate 0.05
```

**内存优化配置（适用于GPU内存有限的情况）:**
```bash
python train.py -s /path/to/dataset -m output/model \
    --use_gt_dca \
    --gt_dca_feature_dim 128 \
    --gt_dca_num_sample_points 4 \
    --gt_dca_attention_heads 4 \
    --gt_dca_dropout_rate 0.2
```

#### GT-DCA性能优化建议

1. **特征维度选择**: 
   - 256维适用于大多数场景
   - 512维用于高质量重建
   - 128维用于快速训练或内存受限环境

2. **采样点数量**:
   - 8个采样点提供良好的质量/性能平衡
   - 16个采样点用于精细细节重建
   - 4个采样点用于快速训练

3. **缓存策略**:
   - 启用缓存可提升训练速度
   - 在内存受限时禁用缓存

### 7. 批量处理

处理多个数据集：

```bash
# 批量GT-DCA训练多个场景
for scene in tandt truck train; do
    python train.py -s data/$scene/train -m output/$scene \
        --use_gt_dca \
        --gt_dca_feature_dim 256 \
        --gt_dca_num_sample_points 8
done

# 批量结合几何约束和GT-DCA训练
for scene in tandt truck train; do
    python train.py -s data/$scene/train -m output/$scene \
        --enable_geometric_constraints \
        --use_gt_dca \
        --multiscale_constraints \
        --adaptive_weighting \
        --gt_dca_feature_dim 256
done

# 批量评估
for scene in tandt truck train; do
    python full_eval.py -m output/$scene --detailed_report
done
```

## 如何贡献 (Contributing)

欢迎任何形式的贡献！如果你发现了 bug 或有功能建议，请随时通过 [GitHub Issues](https://github.com/CPy255/GeoTrack-GS/issues) 提出。

## 致谢 (Acknowledgements)

本项目基于以下出色的工作：
*   [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
*   [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

感谢原作者的杰出贡献。

## 许可证 (License)

本项目根据 [LICENSE.md](LICENSE.md) 文件中的条款进行许可。
