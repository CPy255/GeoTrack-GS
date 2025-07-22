# -*- coding: utf-8 -*-
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from utils.depth_utils import estimate_depth
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

# GeoTrack-GS: 导入几何约束模块
from geometric_constraints import (
    ConstraintConfig,
    TrajectoryManagerImpl,
    EnhancedConstraintEngine,
    EnhancedReprojectionValidator,
    ConstraintResult # 确保导入 ConstraintResult 以便类型检查
)
from utils.config_utils import setup_geometric_constraints_config, print_geometric_constraints_summary

# 导入几何正则化模块
from utils.geometry_regularization import create_geometry_regularizer


def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(args)
    # 如果命令行参数启用了 GT-DCA，则显式启用集成（确保训练阶段使用增强外观特征）
    if getattr(args, 'use_gt_dca', False):
        try:
            gaussians.enable_gt_dca()
        except Exception as e:
            print(f"[GT-DCA] 启用失败，将继续使用标准 SH 特征: {e}")
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # GeoTrack-GS: 初始化几何约束系统
    constraint_system = None
    trajectory_manager = None
    reprojection_validator = None
    
    if hasattr(args, 'enable_geometric_constraints') and args.enable_geometric_constraints:
        try:
            # 加载约束配置
            config_path = getattr(args, 'constraint_config_path', None)
            if config_path and os.path.exists(config_path):
                constraint_config = ConstraintConfig.from_file(config_path)
            else:
                constraint_config = ConstraintConfig()
            
            # 初始化轨迹管理器
            if hasattr(args, 'track_path') and args.track_path and os.path.exists(args.track_path):
                trajectory_manager = TrajectoryManagerImpl(constraint_config)
                trajectories = trajectory_manager.load_trajectories(args.track_path)
                print(f"[GeoTrack-GS] Loaded {len(trajectories)} trajectories from {args.track_path}")
            
            # 初始化约束引擎
            constraint_system = EnhancedConstraintEngine(constraint_config)
            
            # 初始化验证器 (已修复: 传入 constraint_system 而不是 config)
            reprojection_validator = EnhancedReprojectionValidator(
                constraint_engine=constraint_system,
                report_dir=os.path.join(args.model_path, "validation_reports")
            )
            
            print("[GeoTrack-GS] Geometric constraint system initialized successfully")
            
        except Exception as e:
            print(f"[GeoTrack-GS] Failed to initialize geometric constraints: {e}")
            constraint_system = None
            trajectory_manager = None
            reprojection_validator = None
    
    # 初始化几何正则化器
    geometry_regularizer = None
    if opt.geometry_reg_enabled:
        geometry_regularizer = create_geometry_regularizer(opt)
        print(f"[Geometry Regularization] Initialized with weight={opt.geometry_reg_weight}, k_neighbors={opt.geometry_reg_k_neighbors}")


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Task: 修改渲染流程以使用增强的外观特征
        # Task: 确保与现有SH系数的兼容性
        # Requirements: 3.3 - Enhanced rendering with GT-DCA features
        
        # Update GT-DCA cache if needed (for training efficiency)
        if hasattr(gaussians, 'is_gt_dca_enabled') and gaussians.is_gt_dca_enabled():
            # Invalidate cache periodically to ensure fresh features
            if iteration % 100 == 0:
                gaussians.invalidate_gt_dca_cache()
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))

        # GeoTrack-GS: 计算几何约束损失
        geometric_constraint_loss = torch.tensor(0.0, device="cuda")
        constraint_result = None # 确保变量存在
        active_trajectories = [] # 确保变量存在

        if constraint_system is not None and trajectory_manager is not None:
            try:
                # 获取活跃轨迹
                active_trajectories = trajectory_manager.get_active_trajectories()
                
                if len(active_trajectories) > 0:
                    # 获取相机和高斯点
                    cameras = scene.getTrainCameras()
                    gaussian_points = gaussians.get_xyz
                    
                    # 计算自适应权重
                    adaptive_weights = constraint_system.compute_adaptive_weights(
                        active_trajectories, 
                        image_regions=image,
                        iteration=iteration
                    )
                    
                    # 计算重投影约束
                    constraint_result = constraint_system.compute_reprojection_constraints(
                        active_trajectories,
                        cameras,
                        gaussian_points
                    )
                    
                    # 计算多尺度约束
                    multiscale_result = constraint_system.compute_multiscale_constraints(
                        active_trajectories,
                        cameras,
                        scales=[1.0, 0.5, 0.25]
                    )
                    
                    # 组合约束损失
                    geometric_constraint_loss = (
                        constraint_result.loss_value + 
                        multiscale_result.loss_value
                    )
                    
                    # 应用动态权重调度
                    constraint_weight = getattr(args, 'geometric_constraint_weight', 0.1)
                    if iteration < 1000:
                        constraint_weight *= 0.1  # 早期阶段降低权重
                    elif iteration < 5000:
                        constraint_weight *= (0.1 + 0.9 * (iteration - 1000) / 4000)  # 逐渐增加
                    
                    geometric_constraint_loss *= constraint_weight
                    
            except Exception as e:
                if iteration % 100 == 0:  # 避免过多日志
                    print(f"[GeoTrack-GS] Warning: Constraint computation failed at iteration {iteration}: {e}")
                geometric_constraint_loss = torch.tensor(0.0, device="cuda")

        rendered_depth = render_pkg["depth"][0]
        midas_depth = torch.tensor(viewpoint_cam.depth_image).cuda()
        rendered_depth = rendered_depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)

        depth_loss = min(
                             (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
                             (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        )
        
        # GeoTrack-GS: 可选择性地禁用深度损失（用于消融实验）
        if not getattr(args, 'disable_depth_loss', False):
            loss += args.depth_weight * depth_loss
        
        # GeoTrack-GS: 添加几何约束损失
        loss += geometric_constraint_loss
        
        # 添加几何正则化损失
        geometry_reg_loss = torch.tensor(0.0, device="cuda")
        if geometry_regularizer is not None:
            try:
                geometry_reg_loss = geometry_regularizer.compute_anisotropic_regularization_loss(
                    xyz=gaussians.get_xyz,
                    scaling=gaussians.get_scaling,
                    rotation=gaussians.get_rotation,
                    iteration=iteration
                )
                loss += geometry_reg_loss
            except Exception as e:
                if iteration % 1000 == 0:  # 降低日志频率
                    print(f"[Geometry Regularization] Warning: Failed at iteration {iteration}: {e}")
                geometry_reg_loss = torch.tensor(0.0, device="cuda")

        if iteration > args.end_sample_pseudo:
            args.depth_weight = 0.001



        if iteration % args.sample_pseudo_interval == 0 and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            if not pseudo_stack:
                pseudo_stack = scene.getPseudoCameras().copy()
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"], mode='train')

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = (1 - pearson_corrcoef(rendered_depth_pseudo, -midas_depth_pseudo)).mean()

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * args.depth_pseudo_weight * depth_loss_pseudo


        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # GeoTrack-GS: 几何约束验证和日志记录
            if (constraint_system is not None and reprojection_validator is not None and 
                iteration % 100 == 0):  # 每100次迭代验证一次
                try:
                    # 验证几何约束
                    if constraint_result is not None:
                        # 已修复: 传入了正确的参数并解包了返回值
                        is_valid, validation_metrics = reprojection_validator.validate_constraints(
                            constraint_result, 
                            active_trajectories, 
                            iteration
                        )
                        
                        # 记录验证指标到tensorboard
                        if tb_writer:
                            tb_writer.add_scalar('geometric_constraints/constraint_satisfaction', 
                                                 validation_metrics.constraint_satisfaction, iteration)
                            tb_writer.add_scalar('geometric_constraints/geometric_consistency', 
                                                 validation_metrics.geometric_consistency, iteration)
                            # 已修复: 从 constraint_result 获取平均误差
                            tb_writer.add_scalar('geometric_constraints/reprojection_error_mean', 
                                                 constraint_result.mean_error, iteration)
                            tb_writer.add_scalar('geometric_constraints/constraint_loss', 
                                                 geometric_constraint_loss.item(), iteration)
                        
                        # 如果约束验证失败，记录警告
                        if not is_valid:
                            print(f"[GeoTrack-GS] Warning: Constraint validation failed at iteration {iteration}. "
                                  f"Satisfaction: {validation_metrics.constraint_satisfaction:.3f}")
                            
                except Exception as e:
                    if iteration % 500 == 0:  # 减少错误日志频率
                        print(f"[GeoTrack-GS] Warning: Validation failed at iteration {iteration}: {e}")

            # Log and save
            # Task: 确保与现有SH系数的兼容性 - Add GT-DCA performance logging
            gt_dca_info = None
            if hasattr(gaussians, 'get_gt_dca_info'):
                gt_dca_info = gaussians.get_gt_dca_info()
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            testing_iterations, scene, render, (pipe, background), 
                            geometric_constraint_loss if 'geometric_constraint_loss' in locals() else None,
                            gt_dca_info,
                            geometry_reg_loss if 'geometry_reg_loss' in locals() else None)

            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene.cameras_extent, size_threshold, iteration)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians.reset_opacity()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, geometric_constraint_loss=None, gt_dca_info=None, geometry_reg_loss=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # GeoTrack-GS: 记录几何约束损失
        if geometric_constraint_loss is not None:
            tb_writer.add_scalar('train_loss_patches/geometric_constraint_loss', geometric_constraint_loss.item(), iteration)
        
        # 记录几何正则化损失
        if geometry_reg_loss is not None:
            tb_writer.add_scalar('train_loss_patches/geometry_regularization_loss', geometry_reg_loss.item(), iteration)
        
        # Task: 确保与现有SH系数的兼容性 - GT-DCA performance logging
        if gt_dca_info is not None and gt_dca_info.get('status') == 'initialized':
            tb_writer.add_scalar('gt_dca/enabled', 1 if gt_dca_info.get('enabled', False) else 0, iteration)
            
            # Log GT-DCA performance statistics if available
            if 'performance_stats' in gt_dca_info:
                perf_stats = gt_dca_info['performance_stats']
                if 'forward_calls' in perf_stats:
                    tb_writer.add_scalar('gt_dca/forward_calls', perf_stats['forward_calls'], iteration)
                if 'average_processing_time' in perf_stats:
                    tb_writer.add_scalar('gt_dca/avg_processing_time', perf_stats['average_processing_time'], iteration)
            
            # Log GT-DCA configuration
            if 'config' in gt_dca_info:
                config = gt_dca_info['config']
                if hasattr(config, 'feature_dim'):
                    tb_writer.add_scalar('gt_dca/feature_dim', config.feature_dim, iteration)
                if hasattr(config, 'num_sample_points'):
                    tb_writer.add_scalar('gt_dca/num_sample_points', config.num_sample_points, iteration)
        
        # tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 8):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    _mask = None
                    _psnr = psnr(image, gt_image, _mask).mean().double()
                    _ssim = ssim(image, gt_image, _mask).mean().double()
                    _lpips = lpips(image, gt_image, _mask, net_type='vgg')
                    psnr_test += _psnr
                    ssim_test += _ssim
                    lpips_test += _lpips
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} ".format(
                    iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--llff_holdout", type=int, default=0, help="Holdout factor for LLFF data. 1/N of images are used for testing. Default=0 means all for training.")

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000, 5000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--data_type", type=str, default="colmap", help="Type of dataset, e.g., 'colmap', 'blender', '360'")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[5000, 10000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--train_bg", action="store_true")

    # Note: Geometric constraint parameters are already defined in ModelParams and OptimizationParams classes
    
    # Task: 添加GT-DCA启用/禁用的配置选项
    # GT-DCA: Add arguments for GT-DCA appearance modeling
    parser.add_argument("--use_gt_dca", action="store_true", help="Enable GT-DCA enhanced appearance modeling.")
    parser.add_argument("--gt_dca_feature_dim", type=int, default=256, help="GT-DCA feature dimension.")
    parser.add_argument("--gt_dca_num_sample_points", type=int, default=8, help="Number of sampling points for GT-DCA deformable sampling.")
    parser.add_argument("--gt_dca_hidden_dim", type=int, default=128, help="Hidden dimension for GT-DCA MLPs.")
    parser.add_argument("--gt_dca_confidence_threshold", type=float, default=0.5, help="Confidence threshold for GT-DCA track points.")
    parser.add_argument("--gt_dca_min_track_points", type=int, default=4, help="Minimum number of track points required for GT-DCA.")
    parser.add_argument("--gt_dca_enable_caching", action="store_true", help="Enable GT-DCA feature caching for performance.")
    parser.add_argument("--gt_dca_dropout_rate", type=float, default=0.1, help="Dropout rate for GT-DCA modules.")
    parser.add_argument("--gt_dca_attention_heads", type=int, default=8, help="Number of attention heads for GT-DCA cross-attention.")
    # 混合精度选项
    parser.add_argument("--gt_dca_mixed_precision", action="store_true", help="启用 GT-DCA 推理的混合精度 (AMP)")
    parser.add_argument("--gt_dca_amp_dtype", type=str, choices=["fp16", "bf16"], default="fp16", help="AMP 精度类型 (fp16 或 bf16)")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(args.test_iterations)

    # GeoTrack-GS: 设置几何约束配置 (此部分逻辑可被上面直接添加的参数替代，但为保持结构完整性而保留)
    if hasattr(args, 'enable_geometric_constraints') and args.enable_geometric_constraints:
        if not setup_geometric_constraints_config(args):
            print("Failed to setup geometric constraints configuration. Exiting.")
            sys.exit(1)
        
        # GeoTrack-GS: 打印配置摘要
        print_geometric_constraints_summary(args)
    
    # Task: 添加GT-DCA启用/禁用的配置选项
    # GT-DCA: Setup GT-DCA configuration from command line arguments
    if hasattr(args, 'use_gt_dca') and args.use_gt_dca:
        from gt_dca.core.data_structures import GTDCAConfig
        
        # Create GT-DCA configuration optimized for Tesla T4 16GB
        gt_dca_config = GTDCAConfig(
            feature_dim=getattr(args, 'gt_dca_feature_dim', 64),   # Further reduced to 64
            hidden_dim=getattr(args, 'gt_dca_hidden_dim', 32),    # Further reduced to 32
            num_sample_points=getattr(args, 'gt_dca_num_sample_points', 2),  # Minimal sampling points
            confidence_threshold=getattr(args, 'gt_dca_confidence_threshold', 0.8),  # Higher threshold
            min_track_points=getattr(args, 'gt_dca_min_track_points', 4),
            enable_caching=True,  # Enable caching for performance
            dropout_rate=getattr(args, 'gt_dca_dropout_rate', 0.0),  # Disable dropout for speed
            attention_heads=getattr(args, 'gt_dca_attention_heads', 2),  # Minimal attention heads
            use_mixed_precision=getattr(args, 'gt_dca_mixed_precision', False),
            amp_dtype=getattr(args, 'gt_dca_amp_dtype', 'fp16')
        )
        
        # Attach GT-DCA configuration to args
        args.gt_dca_config = gt_dca_config
        
        # GT-DCA: 独立的轨迹文件处理（不依赖几何约束系统）
        if hasattr(args, 'track_path') and args.track_path:
            if os.path.exists(args.track_path):
                print(f"✅ GT-DCA轨迹文件已找到: {args.track_path}")
            else:
                print(f"⚠️ GT-DCA轨迹文件不存在: {args.track_path}")
                print("请确保轨迹文件路径正确，或生成轨迹文件后重新运行")
        else:
            print("⚠️ 未指定GT-DCA轨迹文件路径，请使用 --track_path 参数指定")
        
        print(f"✅ GT-DCA配置已设置: {gt_dca_config}")
    else:
        args.use_gt_dca = False
        args.gt_dca_config = None

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
