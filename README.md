# GeoTrack-GS

本项目利用 3D 高斯溅射（3D Gaussian Splatting）技术，结合先进的几何约束系统，实现高质量的 3D 场景重建与渲染。该项目基于论文《3D Gaussian Splatting for Real-Time Radiance Field Rendering》，并集成了多尺度几何约束、自适应权重调整和 CUDA 加速优化等创新功能。

## 主要功能 (Features)

*   **高质量实时渲染**: 基于 3D Gaussian Splatting，实现照片级的实时渲染效果
*   **几何约束系统**: 集成多尺度几何约束，提升重建精度和一致性
*   **自适应权重调整**: 动态调整约束权重，优化训练过程
*   **轨迹管理**: 智能相机轨迹管理，提升多视角一致性
*   **重投影验证**: 实时重投影误差验证，确保几何准确性
*   **端到端工作流**: 支持从 COLMAP 数据集直接进行训练、渲染和评估
*   **置信度评估**: 集成置信度信息，提升渲染的稳定性和准确性
*   **PyTorch 核心**: 完全使用 PyTorch 构建，易于理解、修改和扩展
*   **高性能 CUDA 加速**: 关键的渲染管线和几何约束计算使用自定义 CUDA 核心实现

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

# 使用 tools/llff.py 脚本处理 LLFF 数据集
python tools/llff.py -s /path/to/your/images

# 指定输出路径
python tools/llff.py -s /path/to/your/images -o /path/to/output

# 高质量 LLFF 处理
python tools/llff.py -s /path/to/your/images -o /path/to/output --quality high --feature_type sift
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

##### 常见数据集格式转换

**从其他格式转换到 COLMAP：**

```bash
# 从 NeRF 格式转换
python convert.py --source_format nerf -s /path/to/nerf/dataset -o /path/to/colmap/output

# 从 Blender 格式转换
python convert.py --source_format blender -s /path/to/blender/dataset -o /path/to/colmap/output

# 从视频提取帧并处理
python convert.py --source_format video -s /path/to/video.mp4 -o /path/to/output --fps 2
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
# 高质量特征提取
python convert.py -s /path/to/images \
    --colmap_executable colmap \
    --camera OPENCV \
    --feature_type sift \
    --quality high \
    --num_threads 8 \
    --gpu_index 0

# 针对困难场景的参数
python convert.py -s /path/to/images \
    --colmap_executable colmap \
    --camera OPENCV \
    --feature_type sift \
    --quality extreme \
    --matcher_type exhaustive \
    --ba_refine_focal_length \
    --ba_refine_principal_point
```

##### 故障排除

**常见问题及解决方案：**

```bash
# 如果 COLMAP 重建失败，尝试降低质量要求
python convert.py -s /path/to/images --quality medium --feature_type orb

# 如果图像过多导致内存不足
python convert.py -s /path/to/images --max_num_images 200 --quality medium

# 如果特征匹配失败
python convert.py -s /path/to/images --matcher_type sequential --overlap 10

# 检查 COLMAP 日志
python convert.py -s /path/to/images --verbose --log_level 2
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

#### CUDA 加速训练
启用 CUDA 优化以提升训练性能：

```bash
# 启用 CUDA 加速
python train.py -s data/tandt/train -m output/tandt_cuda \
    --enable_cuda_optimization \
    --cuda_memory_optimization \
    --performance_monitoring

# 结合几何约束和 CUDA 加速
python train.py -s data/tandt/train -m output/tandt_optimal \
    --enable_geometric_constraints \
    --enable_cuda_optimization \
    --multiscale_constraints \
    --adaptive_weighting \
    --cuda_memory_optimization \
    --constraint_weight 0.12 \
    --iterations 20000
```

#### 训练参数说明
*   `-s, --source_path`: 输入数据集所在的目录路径
*   `-m, --model_path`: 用于保存训练好的模型和检查点的目录路径
*   `--enable_geometric_constraints`: 启用几何约束系统
*   `--multiscale_constraints`: 启用多尺度约束
*   `--adaptive_weighting`: 启用自适应权重调整
*   `--trajectory_management`: 启用轨迹管理
*   `--reprojection_validation`: 启用重投影验证
*   `--constraint_weight`: 几何约束权重 (默认: 0.1)
*   `--enable_cuda_optimization`: 启用 CUDA 优化
*   `--cuda_memory_optimization`: 启用 CUDA 内存优化
*   `--iterations`: 训练迭代次数 (默认: 30000)

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
    --multiscale_constraints \
    --render_quality high
```

#### CUDA 加速渲染
启用 CUDA 优化以提升渲染性能：

```bash
# CUDA 加速渲染
python render.py -m output/tandt_cuda --enable_cuda_optimization

# 完整优化渲染
python render.py -m output/tandt_optimal \
    --enable_geometric_constraints \
    --enable_cuda_optimization \
    --multiscale_constraints \
    --render_quality high \
    --cuda_memory_optimization
```

#### 渲染参数说明
*   `-m, --model_path`: 训练好的模型路径
*   `--enable_geometric_constraints`: 启用几何约束
*   `--reprojection_validation`: 启用重投影验证
*   `--multiscale_constraints`: 启用多尺度约束
*   `--render_quality`: 渲染质量 (low/medium/high)
*   `--enable_cuda_optimization`: 启用 CUDA 优化
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

#### 性能评估
评估 CUDA 优化的性能提升：

```bash
# 性能基准测试
python full_eval.py -m output/tandt_cuda \
    --enable_cuda_optimization \
    --performance_benchmark

# 内存使用评估
python full_eval.py -m output/tandt_optimal \
    --enable_cuda_optimization \
    --memory_profiling \
    --performance_benchmark
```

#### 评估参数说明
*   `-m, --model_path`: 要评估的模型路径
*   `--enable_geometric_constraints`: 启用几何约束评估
*   `--geometric_metrics`: 计算几何相关指标
*   `--reprojection_metrics`: 计算重投影误差指标
*   `--trajectory_metrics`: 计算轨迹一致性指标
*   `--detailed_report`: 生成详细评估报告
*   `--performance_benchmark`: 进行性能基准测试
*   `--memory_profiling`: 进行内存使用分析

### 5. 配置文件

项目支持通过配置文件自定义几何约束参数：

```bash
# 使用自定义配置文件训练
python train.py -s data/tandt/train -m output/tandt_custom \
    --config config/constraints.json

# 查看默认配置
python -c "from geometric_constraints.config import load_config; print(load_config())"
```

### 6. 批量处理

处理多个数据集：

```bash
# 批量训练多个场景
for scene in tandt truck train; do
    python train.py -s data/$scene/train -m output/$scene \
        --enable_geometric_constraints \
        --multiscale_constraints \
        --adaptive_weighting
done

# 批量评估
for scene in tandt truck train; do
    python full_eval.py -m output/$scene \
        --enable_geometric_constraints \
        --geometric_metrics \
        --detailed_report
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
