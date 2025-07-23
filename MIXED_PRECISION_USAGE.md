# 混合精度训练使用指南

## 概述

本项目现已支持完整的混合精度训练（AMP），通过统一的参数控制整个训练流程的精度。

## 参数说明

### 核心参数
- `--mixed_precision`: 启用全局混合精度训练（布尔开关）
- `--amp_dtype`: 混合精度数据类型，可选 `fp16` 或 `bf16`（默认：fp16）

### AMP覆盖范围
启用 `--mixed_precision` 后，以下所有计算都将使用混合精度：

1. **渲染过程**
   - 3D Gaussian渲染 
   - GT-DCA增强外观特征计算
   - CUDA kernel计算

2. **损失计算**
   - L1损失
   - SSIM损失
   - 深度损失（pearson相关系数）
   - 几何约束损失
   - 几何正则化损失

3. **优化过程**
   - 梯度计算和反向传播
   - 优化器参数更新
   - 梯度缩放和裁剪

## 使用示例

### 基础训练（推荐）
```bash
python train.py \
    --source_path ./data/your_scene \
    --model_path ./output \
    --mixed_precision \
    --amp_dtype fp16 \
    --iterations 30000
```

### 高性能训练（GPU内存优化）
```bash
python train.py \
    --source_path ./data/your_scene \
    --model_path ./output \
    --mixed_precision \
    --amp_dtype fp16 \
    --use_gt_dca \
    --gt_dca_feature_dim 128 \
    --gt_dca_num_sample_points 4 \
    --iterations 30000
```

### 使用bf16（现代GPU推荐）
```bash
python train.py \
    --source_path ./data/your_scene \
    --model_path ./output \
    --mixed_precision \
    --amp_dtype bf16 \
    --iterations 30000
```

### 渲染
```bash
python render.py \
    --model_path ./output \
    --mixed_precision \
    --amp_dtype fp16
```

## 性能对比

| 模式 | GPU内存使用 | 训练速度 | 数值稳定性 |
|------|-------------|----------|------------|
| FP32 | 100% | 1.0x | 最佳 |
| FP16 | ~50% | ~1.8x | 良好 |
| BF16 | ~50% | ~1.6x | 更佳 |

## 兼容性说明

### GPU兼容性
- **FP16**: 支持 RTX 20系列及以上、V100及以上
- **BF16**: 支持 RTX 30系列及以上、A100及以上

### 数值精度
- FP16在极端场景下可能出现梯度下溢，但GradScaler会自动处理
- BF16数值范围更大，稳定性更好，但速度稍慢
- 建议优先使用FP16，如遇到训练不稳定再切换到BF16

## 故障排除

### 训练不稳定
```bash
# 尝试使用bf16
--amp_dtype bf16

# 或关闭混合精度
# 移除 --mixed_precision 参数
```

### 内存不足
```bash
# 降低GT-DCA参数
--gt_dca_feature_dim 64
--gt_dca_num_sample_points 2

# 或降低图像分辨率
--resolution 2  # 1/2分辨率
```

### 验证AMP功能
```bash
python test_amp.py  # 测试基本AMP功能
```

## 与旧版本的区别

### ❌ 旧版本（仅GT-DCA混合精度）
```bash
--gt_dca_mixed_precision --gt_dca_amp_dtype fp16
```

### ✅ 新版本（全局混合精度）
```bash
--mixed_precision --amp_dtype fp16
```

新版本的优势：
- 全训练流程AMP加速
- 更大的内存节省
- 统一的参数管理
- 更好的性能表现

## 建议配置

### Tesla T4 (16GB)
```bash
--mixed_precision --amp_dtype fp16 --gt_dca_feature_dim 128
```

### RTX 3080 (10GB) 
```bash
--mixed_precision --amp_dtype fp16 --gt_dca_feature_dim 256
```

### RTX 4090 (24GB)
```bash
--mixed_precision --amp_dtype bf16 --gt_dca_feature_dim 512
```