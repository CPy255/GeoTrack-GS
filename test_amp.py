#!/usr/bin/env python3
"""
测试混合精度训练的简单验证脚本
"""

import torch
from torch.cuda.amp import autocast, GradScaler
import argparse

def test_amp_functionality():
    """测试AMP基本功能"""
    print("🧪 测试AMP基本功能...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法测试AMP")
        return False
    
    # 创建简单的测试数据
    device = torch.device('cuda')
    x = torch.randn(100, 64, requires_grad=True, device=device)
    y = torch.randn(100, 64, device=device)
    
    # 创建简单模型
    model = torch.nn.Linear(64, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()
    
    # 测试FP16训练
    try:
        with autocast(dtype=torch.float16):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        print("✅ AMP基本功能测试通过")
        return True
    except Exception as e:
        print(f"❌ AMP基本功能测试失败: {e}")
        return False

def test_amp_with_gaussian_ops():
    """测试AMP与3D Gaussian相关操作的兼容性"""
    print("🧪 测试AMP与3D Gaussian操作的兼容性...")
    
    try:
        device = torch.device('cuda')
        
        # 模拟3D Gaussian参数
        xyz = torch.randn(1000, 3, device=device, requires_grad=True)
        scaling = torch.randn(1000, 3, device=device, requires_grad=True)
        rotation = torch.randn(1000, 4, device=device, requires_grad=True)
        features = torch.randn(1000, 48, device=device, requires_grad=True)  # SH coefficients
        
        optimizer = torch.optim.Adam([xyz, scaling, rotation, features])
        scaler = GradScaler()
        
        with autocast(dtype=torch.float16):
            # 模拟损失计算
            # 1. L1损失
            rendered = torch.sigmoid(features[:, :3])  # 简化的颜色
            gt = torch.rand_like(rendered)
            l1_loss = torch.abs(rendered - gt).mean()
            
            # 2. 深度损失模拟
            depth_pred = xyz[:, 2]  # 简化的深度
            depth_gt = torch.randn_like(depth_pred)
            # 模拟pearson相关系数计算
            depth_loss = 1 - torch.corrcoef(torch.stack([depth_pred, depth_gt]))[0, 1]
            
            # 3. 几何正则化模拟
            scaling_activated = torch.exp(scaling)
            reg_loss = torch.mean(scaling_activated.max(dim=1)[0] / (scaling_activated.min(dim=1)[0] + 1e-6))
            
            total_loss = l1_loss + 0.03 * depth_loss + 0.01 * reg_loss
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        print("✅ AMP与3D Gaussian操作兼容性测试通过")
        return True
    except Exception as e:
        print(f"❌ AMP与3D Gaussian操作兼容性测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='测试AMP功能')
    args = parser.parse_args()
    
    print("🚀 开始测试AMP实现...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("-" * 50)
    
    all_passed = True
    
    # 基本AMP测试
    if not test_amp_functionality():
        all_passed = False
    
    # 3D Gaussian相关测试
    if not test_amp_with_gaussian_ops():
        all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("🎉 所有AMP测试通过！您的实现应该可以正常工作。")
        print("💡 建议：在实际训练中使用 --mixed_precision 参数启用AMP")
    else:
        print("⚠️ 部分测试失败，请检查CUDA环境和AMP支持")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)