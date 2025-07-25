#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练参数传递是否正确
Test Train Parameters Script
"""

import sys
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def test_parameter_passing():
    """测试参数传递是否正确"""
    print("🧪 测试训练参数传递...")
    
    # 模拟命令行参数
    test_args = [
        "--source_path", "/workspace/llff/nerf_llff_data/flower",
        "--model_path", "output/test_gtdca",
        "--use_gt_dca",
        "--enable_geometric_constraints", 
        "--enable_geometry_regularization",
        "--track_path", "tracks.h5",
        "--iteration", "1000"
    ]
    
    # 创建参数解析器
    parser = ArgumentParser(description="Parameter Test")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # 添加训练脚本中的自定义参数
    parser.add_argument("--use_gt_dca", action="store_true", help="Enable GT-DCA enhanced appearance modeling.")
    parser.add_argument("--enable_geometric_constraints", action="store_true", help="Enable geometric constraints system.")
    parser.add_argument("--enable_geometry_regularization", action="store_true", help="Enable geometry regularization.")
    parser.add_argument("--track_path", type=str, default="", help="Path to trajectory file (tracks.h5).")
    
    # 解析参数
    args = parser.parse_args(test_args)
    
    # 处理快捷启用参数
    if hasattr(args, 'enable_geometry_regularization') and args.enable_geometry_regularization:
        args.geometry_reg_enabled = True
    
    # 提取参数组
    lp_extracted = lp.extract(args)
    op_extracted = op.extract(args)
    pp_extracted = pp.extract(args)
    
    # 测试结果
    print("\n📊 参数提取结果:")
    print(f"ModelParams:")
    print(f"  - source_path: {getattr(lp_extracted, 'source_path', 'NOT_FOUND')}")
    print(f"  - model_path: {getattr(lp_extracted, 'model_path', 'NOT_FOUND')}")
    print(f"  - enable_geometric_constraints: {getattr(lp_extracted, 'enable_geometric_constraints', 'NOT_FOUND')}")
    print(f"  - track_path: {getattr(lp_extracted, 'track_path', 'NOT_FOUND')}")
    
    print(f"\nOptimizationParams:")
    print(f"  - iterations: {getattr(op_extracted, 'iterations', 'NOT_FOUND')}")
    print(f"  - geometry_reg_enabled: {getattr(op_extracted, 'geometry_reg_enabled', 'NOT_FOUND')}")
    print(f"  - geometry_reg_weight: {getattr(op_extracted, 'geometry_reg_weight', 'NOT_FOUND')}")
    print(f"  - geometric_constraint_weight: {getattr(op_extracted, 'geometric_constraint_weight', 'NOT_FOUND')}")
    
    print(f"\n原始args:")
    print(f"  - use_gt_dca: {getattr(args, 'use_gt_dca', 'NOT_FOUND')}")
    print(f"  - enable_geometric_constraints: {getattr(args, 'enable_geometric_constraints', 'NOT_FOUND')}")
    print(f"  - enable_geometry_regularization: {getattr(args, 'enable_geometry_regularization', 'NOT_FOUND')}")
    print(f"  - geometry_reg_enabled: {getattr(args, 'geometry_reg_enabled', 'NOT_FOUND')}")
    
    # 验证关键参数
    success = True
    
    if not getattr(args, 'use_gt_dca', False):
        print("❌ GT-DCA参数未正确设置")
        success = False
    
    if not getattr(args, 'enable_geometric_constraints', False):
        print("❌ 几何约束参数未正确设置")
        success = False
        
    if not getattr(op_extracted, 'geometry_reg_enabled', False):
        print("❌ 几何正则化参数未正确设置")
        success = False
    
    if success:
        print("\n✅ 所有参数传递测试通过！")
    else:
        print("\n❌ 参数传递测试失败！")
    
    return success

if __name__ == "__main__":
    test_parameter_passing()