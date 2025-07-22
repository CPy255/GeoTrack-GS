"""
自适应权重系统

实现基于纹理、置信度和训练阶段的自适应权重计算
"""

from typing import List, Optional, Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from .data_structures import Trajectory
from .config import ConstraintConfig


class AdaptiveWeighting:
    """自适应权重计算器"""
    
    def __init__(self, config: ConstraintConfig):
        """
        初始化自适应权重计算器
        
        Args:
            config: 约束系统配置
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_torch_dtype()
        
        # 缓存计算结果
        self._texture_cache = {}
        self._weight_history = []
    
    def compute_texture_weights(self, 
                              image_regions: torch.Tensor,
                              trajectory_points: torch.Tensor) -> torch.Tensor:
        """
        计算基于纹理的权重
        
        Args:
            image_regions: 图像区域张量 [H, W, C]
            trajectory_points: 轨迹点坐标 [N, 2]
            
        Returns:
            纹理权重张量 [N]
        """
        if image_regions is None or trajectory_points.size(0) == 0:
            return torch.ones(trajectory_points.size(0), 
                            device=self.device, dtype=self.dtype)
        
        # 确保输入在正确设备上
        image_regions = image_regions.to(self.device)
        trajectory_points = trajectory_points.to(self.device)
        
        # 计算图像梯度
        gray_image = self._convert_to_grayscale(image_regions)
        gradients = self._compute_image_gradients(gray_image)
        
        # 在轨迹点位置采样梯度强度
        gradient_magnitudes = self._sample_gradients_at_points(
            gradients, trajectory_points)
        
        # 将梯度强度转换为权重
        texture_weights = self._gradients_to_weights(gradient_magnitudes)
        
        return texture_weights   
 def compute_confidence_weights(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """
        计算基于置信度的权重
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            置信度权重张量 [N]
        """
        if not trajectories:
            return torch.tensor([], device=self.device, dtype=self.dtype)
        
        confidence_scores = []
        for trajectory in trajectories:
            avg_confidence = trajectory.average_confidence
            confidence_scores.append(avg_confidence)
        
        confidence_tensor = torch.tensor(confidence_scores, 
                                       device=self.device, dtype=self.dtype)
        
        # 应用衰减因子
        decay_factor = self.config.weighting.confidence_decay_factor
        confidence_weights = torch.pow(confidence_tensor, decay_factor)
        
        # 应用最小权重约束
        min_weight = self.config.weighting.min_confidence_weight
        confidence_weights = torch.clamp(confidence_weights, min=min_weight)
        
        return confidence_weights
    
    def compute_temporal_weights(self, iteration: int) -> float:
        """
        计算基于训练阶段的时序权重
        
        Args:
            iteration: 当前训练迭代次数
            
        Returns:
            时序权重标量
        """
        early_iters = self.config.weighting.early_stage_iterations
        mid_iters = self.config.weighting.mid_stage_iterations
        
        if iteration < early_iters:
            return self.config.weighting.early_stage_weight
        elif iteration < mid_iters:
            # 线性插值
            progress = (iteration - early_iters) / (mid_iters - early_iters)
            early_weight = self.config.weighting.early_stage_weight
            mid_weight = self.config.weighting.mid_stage_weight
            return early_weight + progress * (mid_weight - early_weight)
        else:
            return self.config.weighting.final_stage_weight
    
    def compute_adaptive_weights(self,
                               trajectories: List[Trajectory],
                               image_regions: Optional[torch.Tensor] = None,
                               iteration: int = 0) -> torch.Tensor:
        """
        计算综合自适应权重
        
        Args:
            trajectories: 轨迹列表
            image_regions: 图像区域张量
            iteration: 当前训练迭代次数
            
        Returns:
            综合权重张量 [N]
        """
        if not trajectories:
            return torch.tensor([], device=self.device, dtype=self.dtype)
        
        # 获取轨迹点坐标
        trajectory_points = self._extract_trajectory_points(trajectories)
        
        # 计算各种权重
        texture_weights = self.compute_texture_weights(image_regions, trajectory_points)
        confidence_weights = self.compute_confidence_weights(trajectories)
        temporal_weight = self.compute_temporal_weights(iteration)
        
        # 组合权重
        combined_weights = texture_weights * confidence_weights * temporal_weight
        
        # 归一化权重
        normalized_weights = self._normalize_weights(combined_weights)
        
        # 更新权重历史
        self._update_weight_history(normalized_weights, iteration)
        
        return normalized_weights    def _c
onvert_to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """将图像转换为灰度"""
        if image.dim() == 3 and image.size(2) == 3:
            # RGB to grayscale
            weights = torch.tensor([0.299, 0.587, 0.114], 
                                 device=image.device, dtype=image.dtype)
            return torch.sum(image * weights, dim=2)
        elif image.dim() == 3 and image.size(2) == 1:
            return image.squeeze(2)
        else:
            return image
    
    def _compute_image_gradients(self, image: torch.Tensor) -> torch.Tensor:
        """计算图像梯度"""
        # 添加批次维度
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              device=image.device, dtype=image.dtype).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              device=image.device, dtype=image.dtype).unsqueeze(0).unsqueeze(0)
        
        # 计算梯度
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        
        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_magnitude.squeeze()
    
    def _sample_gradients_at_points(self, 
                                  gradients: torch.Tensor, 
                                  points: torch.Tensor) -> torch.Tensor:
        """在指定点采样梯度值"""
        H, W = gradients.shape
        
        # 将点坐标归一化到[-1, 1]范围
        normalized_points = points.clone()
        normalized_points[:, 0] = 2.0 * points[:, 0] / W - 1.0  # x坐标
        normalized_points[:, 1] = 2.0 * points[:, 1] / H - 1.0  # y坐标
        
        # 重塑为grid_sample所需格式
        grid = normalized_points.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        gradients_batch = gradients.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # 双线性插值采样
        sampled_gradients = F.grid_sample(
            gradients_batch, grid, 
            mode='bilinear', padding_mode='border', align_corners=True)
        
        return sampled_gradients.squeeze()
    
    def _gradients_to_weights(self, gradient_magnitudes: torch.Tensor) -> torch.Tensor:
        """将梯度强度转换为权重"""
        threshold = self.config.weighting.texture_gradient_threshold
        min_weight = self.config.weighting.texture_weight_min
        max_weight = self.config.weighting.texture_weight_max
        
        # 使用sigmoid函数进行平滑映射
        normalized_gradients = gradient_magnitudes / threshold
        sigmoid_weights = torch.sigmoid(normalized_gradients - 1.0)
        
        # 映射到指定权重范围
        texture_weights = min_weight + sigmoid_weights * (max_weight - min_weight)
        
        return texture_weights 
   def _extract_trajectory_points(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """提取轨迹点坐标"""
        all_points = []
        for trajectory in trajectories:
            points_tensor = trajectory.get_points_tensor()
            all_points.append(points_tensor)
        
        if all_points:
            return torch.cat(all_points, dim=0).to(self.device)
        else:
            return torch.empty((0, 2), device=self.device, dtype=self.dtype)
    
    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """归一化权重"""
        if weights.numel() == 0:
            return weights
        
        # 避免除零
        weight_sum = weights.sum()
        if weight_sum > 1e-8:
            return weights / weight_sum * weights.numel()
        else:
            return torch.ones_like(weights)
    
    def _update_weight_history(self, weights: torch.Tensor, iteration: int):
        """更新权重历史记录"""
        weight_stats = {
            'iteration': iteration,
            'mean_weight': float(weights.mean()) if weights.numel() > 0 else 0.0,
            'std_weight': float(weights.std()) if weights.numel() > 0 else 0.0,
            'min_weight': float(weights.min()) if weights.numel() > 0 else 0.0,
            'max_weight': float(weights.max()) if weights.numel() > 0 else 0.0,
        }
        
        self._weight_history.append(weight_stats)
        
        # 限制历史记录长度
        max_history = 1000
        if len(self._weight_history) > max_history:
            self._weight_history = self._weight_history[-max_history:]
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """获取权重统计信息"""
        if not self._weight_history:
            return {}
        
        recent_stats = self._weight_history[-1]
        return {
            'current_mean_weight': recent_stats['mean_weight'],
            'current_std_weight': recent_stats['std_weight'],
            'current_weight_range': recent_stats['max_weight'] - recent_stats['min_weight'],
            'history_length': len(self._weight_history)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._texture_cache.clear()
        self._weight_history.clear()    
def compute_reprojection_confidence(self, 
                                      reprojection_errors: torch.Tensor,
                                      error_threshold: float = 2.0) -> torch.Tensor:
        """
        基于重投影误差计算置信度
        
        Args:
            reprojection_errors: 重投影误差张量 [N]
            error_threshold: 误差阈值
            
        Returns:
            置信度张量 [N]
        """
        # 使用指数衰减函数将误差转换为置信度
        confidence = torch.exp(-reprojection_errors / error_threshold)
        return confidence
    
    def detect_outliers_by_confidence(self, 
                                    confidence_scores: torch.Tensor,
                                    threshold: float = 0.3) -> torch.Tensor:
        """
        基于置信度检测异常值
        
        Args:
            confidence_scores: 置信度分数 [N]
            threshold: 置信度阈值
            
        Returns:
            异常值掩码 [N] (True表示异常值)
        """
        return confidence_scores < threshold
    
    def suppress_outlier_weights(self, 
                               weights: torch.Tensor,
                               outlier_mask: torch.Tensor,
                               suppression_factor: float = 0.1) -> torch.Tensor:
        """
        抑制异常值的权重
        
        Args:
            weights: 原始权重 [N]
            outlier_mask: 异常值掩码 [N]
            suppression_factor: 抑制因子
            
        Returns:
            调整后的权重 [N]
        """
        adjusted_weights = weights.clone()
        adjusted_weights[outlier_mask] *= suppression_factor
        return adjusted_weights
    
    def update_confidence_dynamically(self, 
                                    trajectories: List[Trajectory],
                                    reprojection_errors: torch.Tensor,
                                    learning_rate: float = 0.1) -> List[Trajectory]:
        """
        动态更新轨迹的置信度分数
        
        Args:
            trajectories: 轨迹列表
            reprojection_errors: 重投影误差 [N]
            learning_rate: 学习率
            
        Returns:
            更新后的轨迹列表
        """
        if len(trajectories) != reprojection_errors.size(0):
            raise ValueError("轨迹数量与误差数量不匹配")
        
        # 计算新的置信度
        new_confidences = self.compute_reprojection_confidence(reprojection_errors)
        
        # 更新轨迹置信度
        updated_trajectories = []
        for i, trajectory in enumerate(trajectories):
            updated_trajectory = trajectory
            current_confidence = trajectory.average_confidence
            new_confidence = float(new_confidences[i])
            
            # 使用指数移动平均更新置信度
            updated_confidence = (1 - learning_rate) * current_confidence + learning_rate * new_confidence
            
            # 更新轨迹中所有点的置信度
            for j, point in enumerate(updated_trajectory.points_2d):
                point.detection_confidence = updated_confidence
            
            # 重新计算置信度分数列表
            updated_trajectory.confidence_scores = [updated_confidence] * len(updated_trajectory.points_2d)
            
            updated_trajectories.append(updated_trajectory)
        
        return updated_trajectories    def cr
eate_weight_scheduler(self, 
                              total_iterations: int,
                              schedule_type: str = "cosine") -> Dict[str, float]:
        """
        创建权重调度器
        
        Args:
            total_iterations: 总训练迭代次数
            schedule_type: 调度类型 ("linear", "cosine", "exponential")
            
        Returns:
            调度参数字典
        """
        scheduler_config = {
            'total_iterations': total_iterations,
            'schedule_type': schedule_type,
            'early_stage_ratio': 0.2,  # 前20%为早期阶段
            'mid_stage_ratio': 0.6,    # 中间60%为中期阶段
            'final_stage_ratio': 0.2   # 最后20%为最终阶段
        }
        
        return scheduler_config
    
    def compute_scheduled_temporal_weight(self, 
                                        iteration: int,
                                        scheduler_config: Dict[str, float]) -> float:
        """
        根据调度策略计算时序权重
        
        Args:
            iteration: 当前迭代次数
            scheduler_config: 调度器配置
            
        Returns:
            调度后的时序权重
        """
        total_iters = scheduler_config['total_iterations']
        schedule_type = scheduler_config['schedule_type']
        
        # 计算进度比例
        progress = min(iteration / total_iters, 1.0)
        
        early_weight = self.config.weighting.early_stage_weight
        mid_weight = self.config.weighting.mid_stage_weight
        final_weight = self.config.weighting.final_stage_weight
        
        if schedule_type == "linear":
            return self._linear_schedule(progress, early_weight, mid_weight, final_weight)
        elif schedule_type == "cosine":
            return self._cosine_schedule(progress, early_weight, final_weight)
        elif schedule_type == "exponential":
            return self._exponential_schedule(progress, early_weight, final_weight)
        else:
            # 默认使用原有的阶段式调度
            return self.compute_temporal_weights(iteration)
    
    def _linear_schedule(self, progress: float, early_weight: float, 
                        mid_weight: float, final_weight: float) -> float:
        """线性调度"""
        if progress < 0.2:
            # 早期阶段：从early_weight线性增长到mid_weight
            local_progress = progress / 0.2
            return early_weight + local_progress * (mid_weight - early_weight)
        elif progress < 0.8:
            # 中期阶段：从mid_weight线性增长到final_weight
            local_progress = (progress - 0.2) / 0.6
            return mid_weight + local_progress * (final_weight - mid_weight)
        else:
            # 最终阶段：保持final_weight
            return final_weight
    
    def _cosine_schedule(self, progress: float, early_weight: float, final_weight: float) -> float:
        """余弦调度"""
        import math
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return final_weight + (early_weight - final_weight) * cosine_factor
    
    def _exponential_schedule(self, progress: float, early_weight: float, final_weight: float) -> float:
        """指数调度"""
        import math
        exp_factor = math.exp(-5 * progress)  # 指数衰减
        return final_weight + (early_weight - final_weight) * exp_factor
    
    def detect_loss_convergence(self, 
                              loss_history: List[float],
                              window_size: int = 100,
                              convergence_threshold: float = 1e-4) -> bool:
        """
        检测损失收敛状态
        
        Args:
            loss_history: 损失历史记录
            window_size: 滑动窗口大小
            convergence_threshold: 收敛阈值
            
        Returns:
            是否已收敛
        """
        if len(loss_history) < window_size:
            return False
        
        # 计算最近window_size个损失值的变化
        recent_losses = loss_history[-window_size:]
        loss_variance = np.var(recent_losses)
        
        return loss_variance < convergence_threshold
    
    def adaptive_weight_adjustment(self, 
                                 current_weights: torch.Tensor,
                                 loss_history: List[float],
                                 performance_metrics: Dict[str, float],
                                 adjustment_factor: float = 0.1) -> torch.Tensor:
        """
        基于性能指标自适应调整权重参数
        
        Args:
            current_weights: 当前权重
            loss_history: 损失历史
            performance_metrics: 性能指标
            adjustment_factor: 调整因子
            
        Returns:
            调整后的权重
        """
        adjusted_weights = current_weights.clone()
        
        # 如果损失已收敛，增加权重以提高约束强度
        if self.detect_loss_convergence(loss_history):
            adjusted_weights *= (1.0 + adjustment_factor)
        
        # 根据几何质量指标调整权重
        if 'geometric_consistency' in performance_metrics:
            consistency = performance_metrics['geometric_consistency']
            if consistency < 0.8:  # 几何一致性较低
                adjusted_weights *= (1.0 + adjustment_factor * 0.5)
        
        # 根据异常值比例调整权重
        if 'outlier_ratio' in performance_metrics:
            outlier_ratio = performance_metrics['outlier_ratio']
            if outlier_ratio > 0.3:  # 异常值比例过高
                adjusted_weights *= (1.0 - adjustment_factor * 0.3)
        
        return adjusted_weights
    
    def get_training_stage_info(self, iteration: int) -> Dict[str, any]:
        """
        获取当前训练阶段信息
        
        Args:
            iteration: 当前迭代次数
            
        Returns:
            训练阶段信息字典
        """
        early_iters = self.config.weighting.early_stage_iterations
        mid_iters = self.config.weighting.mid_stage_iterations
        
        if iteration < early_iters:
            stage = "early"
            stage_progress = iteration / early_iters
        elif iteration < mid_iters:
            stage = "middle"
            stage_progress = (iteration - early_iters) / (mid_iters - early_iters)
        else:
            stage = "final"
            stage_progress = 1.0
        
        return {
            'stage': stage,
            'stage_progress': stage_progress,
            'current_weight': self.compute_temporal_weights(iteration),
            'iteration': iteration
        }
    
    def create_weight_scheduler(self, 
                              total_iterations: int,
                              schedule_type: str = "adaptive") -> 'TrainingWeightScheduler':
        """
        创建训练阶段权重调度器
        
        Args:
            total_iterations: 总训练迭代次数
            schedule_type: 调度类型 ("linear", "cosine", "exponential", "adaptive")
            
        Returns:
            训练权重调度器实例
        """
        return TrainingWeightScheduler(
            config=self.config,
            total_iterations=total_iterations,
            schedule_type=schedule_type,
            device=self.device
        )


class TrainingWeightScheduler:
    """训练阶段权重调度器
    
    实现基于迭代次数的权重调度、损失收敛检测和自适应权重调整策略
    """
    
    def __init__(self, 
                 config: ConstraintConfig,
                 total_iterations: int,
                 schedule_type: str = "adaptive",
                 device: torch.device = None):
        """
        初始化训练权重调度器
        
        Args:
            config: 约束系统配置
            total_iterations: 总训练迭代次数
            schedule_type: 调度类型
            device: 计算设备
        """
        self.config = config
        self.total_iterations = total_iterations
        self.schedule_type = schedule_type
        self.device = device or config.get_device()
        
        # 损失收敛检测参数
        self.loss_history = []
        self.convergence_window_size = 100
        self.convergence_threshold = 1e-4
        self.convergence_patience = 50
        self.no_improvement_count = 0
        self.best_loss = float('inf')
        
        # 自适应调整参数
        self.adjustment_history = []
        self.base_adjustment_factor = 0.1
        self.max_adjustment_factor = 0.5
        self.min_adjustment_factor = 0.01
        
        # 权重调度状态
        self.current_stage = "early"
        self.stage_transition_points = self._compute_stage_transitions()
        self.weight_multipliers = self._initialize_weight_multipliers()
        
        # 性能监控
        self.performance_history = []
        self.last_validation_iteration = 0
        
    def _compute_stage_transitions(self) -> Dict[str, int]:
        """计算训练阶段转换点"""
        early_ratio = 0.2  # 前20%为早期阶段
        mid_ratio = 0.6    # 中间60%为中期阶段
        
        return {
            'early_to_mid': int(self.total_iterations * early_ratio),
            'mid_to_final': int(self.total_iterations * (early_ratio + mid_ratio))
        }
    
    def _initialize_weight_multipliers(self) -> Dict[str, float]:
        """初始化权重乘数"""
        return {
            'texture_multiplier': 1.0,
            'confidence_multiplier': 1.0,
            'temporal_multiplier': 1.0,
            'geometric_multiplier': 1.0
        }
    
    def update_loss_history(self, loss_value: float, iteration: int):
        """
        更新损失历史记录
        
        Args:
            loss_value: 当前损失值
            iteration: 当前迭代次数
        """
        self.loss_history.append({
            'iteration': iteration,
            'loss': loss_value,
            'timestamp': iteration
        })
        
        # 限制历史记录长度
        max_history = 2000
        if len(self.loss_history) > max_history:
            self.loss_history = self.loss_history[-max_history:]
        
        # 更新最佳损失和无改善计数
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
    
    def detect_convergence(self) -> Dict[str, any]:
        """
        检测损失收敛状态
        
        Returns:
            收敛检测结果字典
        """
        if len(self.loss_history) < self.convergence_window_size:
            return {
                'converged': False,
                'convergence_type': None,
                'confidence': 0.0,
                'trend': 'insufficient_data'
            }
        
        recent_losses = [entry['loss'] for entry in self.loss_history[-self.convergence_window_size:]]
        
        # 方差收敛检测
        loss_variance = np.var(recent_losses)
        variance_converged = loss_variance < self.convergence_threshold
        
        # 趋势收敛检测
        trend_slope = self._compute_loss_trend(recent_losses)
        trend_converged = abs(trend_slope) < self.convergence_threshold * 10
        
        # 停滞检测
        stagnation_detected = self.no_improvement_count > self.convergence_patience
        
        # 综合判断
        converged = variance_converged or (trend_converged and len(recent_losses) > 50)
        
        convergence_type = None
        if variance_converged:
            convergence_type = 'variance'
        elif trend_converged:
            convergence_type = 'trend'
        elif stagnation_detected:
            convergence_type = 'stagnation'
        
        # 计算收敛置信度
        confidence = self._compute_convergence_confidence(recent_losses, trend_slope, loss_variance)
        
        return {
            'converged': converged,
            'convergence_type': convergence_type,
            'confidence': confidence,
            'trend': 'decreasing' if trend_slope < 0 else 'increasing' if trend_slope > 0 else 'stable',
            'variance': loss_variance,
            'no_improvement_count': self.no_improvement_count
        }
    
    def _compute_loss_trend(self, losses: List[float]) -> float:
        """计算损失趋势斜率"""
        if len(losses) < 2:
            return 0.0
        
        x = np.arange(len(losses))
        y = np.array(losses)
        
        # 线性回归计算斜率
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _compute_convergence_confidence(self, losses: List[float], trend_slope: float, variance: float) -> float:
        """计算收敛置信度"""
        # 基于方差的置信度
        variance_confidence = max(0.0, 1.0 - variance / self.convergence_threshold)
        
        # 基于趋势的置信度
        trend_confidence = max(0.0, 1.0 - abs(trend_slope) / (self.convergence_threshold * 10))
        
        # 基于稳定性的置信度
        recent_std = np.std(losses[-20:]) if len(losses) >= 20 else np.std(losses)
        stability_confidence = max(0.0, 1.0 - recent_std / np.mean(losses))
        
        # 综合置信度
        overall_confidence = (variance_confidence + trend_confidence + stability_confidence) / 3.0
        return min(1.0, max(0.0, overall_confidence))
    
    def compute_scheduled_weights(self, 
                                iteration: int,
                                performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        计算调度后的权重
        
        Args:
            iteration: 当前迭代次数
            performance_metrics: 性能指标
            
        Returns:
            调度后的权重字典
        """
        # 更新当前训练阶段
        self._update_current_stage(iteration)
        
        # 基础权重调度
        base_weights = self._compute_base_scheduled_weights(iteration)
        
        # 自适应调整
        if performance_metrics:
            adaptive_weights = self._apply_adaptive_adjustments(base_weights, performance_metrics, iteration)
        else:
            adaptive_weights = base_weights
        
        # 收敛状态调整
        convergence_info = self.detect_convergence()
        final_weights = self._apply_convergence_adjustments(adaptive_weights, convergence_info)
        
        # 记录调整历史
        self._record_weight_adjustment(iteration, base_weights, final_weights, convergence_info)
        
        return final_weights
    
    def _update_current_stage(self, iteration: int):
        """更新当前训练阶段"""
        if iteration < self.stage_transition_points['early_to_mid']:
            self.current_stage = "early"
        elif iteration < self.stage_transition_points['mid_to_final']:
            self.current_stage = "middle"
        else:
            self.current_stage = "final"
    
    def _compute_base_scheduled_weights(self, iteration: int) -> Dict[str, float]:
        """计算基础调度权重"""
        progress = min(iteration / self.total_iterations, 1.0)
        
        if self.schedule_type == "linear":
            temporal_weight = self._linear_weight_schedule(progress)
        elif self.schedule_type == "cosine":
            temporal_weight = self._cosine_weight_schedule(progress)
        elif self.schedule_type == "exponential":
            temporal_weight = self._exponential_weight_schedule(progress)
        elif self.schedule_type == "adaptive":
            temporal_weight = self._adaptive_weight_schedule(progress, iteration)
        else:
            temporal_weight = self._default_weight_schedule(iteration)
        
        return {
            'temporal_weight': temporal_weight,
            'texture_weight_multiplier': self.weight_multipliers['texture_multiplier'],
            'confidence_weight_multiplier': self.weight_multipliers['confidence_multiplier'],
            'geometric_weight_multiplier': self.weight_multipliers['geometric_multiplier']
        }
    
    def _linear_weight_schedule(self, progress: float) -> float:
        """线性权重调度"""
        early_weight = self.config.weighting.early_stage_weight
        final_weight = self.config.weighting.final_stage_weight
        
        return early_weight + progress * (final_weight - early_weight)
    
    def _cosine_weight_schedule(self, progress: float) -> float:
        """余弦权重调度"""
        import math
        early_weight = self.config.weighting.early_stage_weight
        final_weight = self.config.weighting.final_stage_weight
        
        cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
        return early_weight + cosine_factor * (final_weight - early_weight)
    
    def _exponential_weight_schedule(self, progress: float) -> float:
        """指数权重调度"""
        import math
        early_weight = self.config.weighting.early_stage_weight
        final_weight = self.config.weighting.final_stage_weight
        
        exp_factor = 1 - math.exp(-3 * progress)
        return early_weight + exp_factor * (final_weight - early_weight)
    
    def _adaptive_weight_schedule(self, progress: float, iteration: int) -> float:
        """自适应权重调度"""
        # 基础余弦调度
        base_weight = self._cosine_weight_schedule(progress)
        
        # 根据损失收敛状态调整
        convergence_info = self.detect_convergence()
        if convergence_info['converged'] and convergence_info['confidence'] > 0.8:
            # 已收敛，增加权重强度
            base_weight *= 1.2
        elif convergence_info['trend'] == 'increasing':
            # 损失上升，降低权重
            base_weight *= 0.8
        
        return base_weight
    
    def _default_weight_schedule(self, iteration: int) -> float:
        """默认阶段式权重调度"""
        early_iters = self.config.weighting.early_stage_iterations
        mid_iters = self.config.weighting.mid_stage_iterations
        
        if iteration < early_iters:
            return self.config.weighting.early_stage_weight
        elif iteration < mid_iters:
            progress = (iteration - early_iters) / (mid_iters - early_iters)
            early_weight = self.config.weighting.early_stage_weight
            mid_weight = self.config.weighting.mid_stage_weight
            return early_weight + progress * (mid_weight - early_weight)
        else:
            return self.config.weighting.final_stage_weight
    
    def _apply_adaptive_adjustments(self, 
                                  base_weights: Dict[str, float],
                                  performance_metrics: Dict[str, float],
                                  iteration: int) -> Dict[str, float]:
        """应用自适应调整"""
        adjusted_weights = base_weights.copy()
        
        # 根据几何一致性调整
        if 'geometric_consistency' in performance_metrics:
            consistency = performance_metrics['geometric_consistency']
            if consistency < 0.8:
                # 几何一致性低，增加几何约束权重
                adjustment = (0.8 - consistency) * self.base_adjustment_factor
                adjusted_weights['geometric_weight_multiplier'] *= (1.0 + adjustment)
        
        # 根据异常值比例调整
        if 'outlier_ratio' in performance_metrics:
            outlier_ratio = performance_metrics['outlier_ratio']
            if outlier_ratio > 0.3:
                # 异常值过多，降低置信度权重
                adjustment = (outlier_ratio - 0.3) * self.base_adjustment_factor
                adjusted_weights['confidence_weight_multiplier'] *= (1.0 - adjustment)
        
        # 根据重投影误差调整
        if 'mean_reprojection_error' in performance_metrics:
            error = performance_metrics['mean_reprojection_error']
            if error > 2.0:
                # 重投影误差大，增加纹理权重
                adjustment = min((error - 2.0) / 5.0, 0.3) * self.base_adjustment_factor
                adjusted_weights['texture_weight_multiplier'] *= (1.0 + adjustment)
        
        return adjusted_weights
    
    def _apply_convergence_adjustments(self, 
                                     weights: Dict[str, float],
                                     convergence_info: Dict[str, any]) -> Dict[str, float]:
        """应用收敛状态调整"""
        adjusted_weights = weights.copy()
        
        if convergence_info['converged']:
            if convergence_info['convergence_type'] == 'stagnation':
                # 训练停滞，增加扰动
                perturbation = 0.1 * (1.0 + 0.1 * np.random.randn())
                for key in adjusted_weights:
                    if 'multiplier' in key:
                        adjusted_weights[key] *= perturbation
            elif convergence_info['confidence'] > 0.9:
                # 高置信度收敛，稳定权重
                for key in adjusted_weights:
                    if 'multiplier' in key:
                        adjusted_weights[key] = 0.9 * adjusted_weights[key] + 0.1 * 1.0
        
        return adjusted_weights
    
    def _record_weight_adjustment(self, 
                                iteration: int,
                                base_weights: Dict[str, float],
                                final_weights: Dict[str, float],
                                convergence_info: Dict[str, any]):
        """记录权重调整历史"""
        adjustment_record = {
            'iteration': iteration,
            'stage': self.current_stage,
            'base_weights': base_weights.copy(),
            'final_weights': final_weights.copy(),
            'convergence_info': convergence_info,
            'adjustment_magnitude': self._compute_adjustment_magnitude(base_weights, final_weights)
        }
        
        self.adjustment_history.append(adjustment_record)
        
        # 限制历史记录长度
        max_history = 1000
        if len(self.adjustment_history) > max_history:
            self.adjustment_history = self.adjustment_history[-max_history:]
    
    def _compute_adjustment_magnitude(self, 
                                    base_weights: Dict[str, float],
                                    final_weights: Dict[str, float]) -> float:
        """计算调整幅度"""
        total_adjustment = 0.0
        count = 0
        
        for key in base_weights:
            if key in final_weights:
                adjustment = abs(final_weights[key] - base_weights[key]) / max(base_weights[key], 1e-8)
                total_adjustment += adjustment
                count += 1
        
        return total_adjustment / max(count, 1)
    
    def get_scheduler_status(self) -> Dict[str, any]:
        """获取调度器状态信息"""
        convergence_info = self.detect_convergence()
        
        return {
            'current_stage': self.current_stage,
            'schedule_type': self.schedule_type,
            'total_iterations': self.total_iterations,
            'loss_history_length': len(self.loss_history),
            'convergence_info': convergence_info,
            'weight_multipliers': self.weight_multipliers.copy(),
            'adjustment_history_length': len(self.adjustment_history),
            'best_loss': self.best_loss,
            'no_improvement_count': self.no_improvement_count
        }
    
    def reset_scheduler(self):
        """重置调度器状态"""
        self.loss_history.clear()
        self.adjustment_history.clear()
        self.performance_history.clear()
        self.no_improvement_count = 0
        self.best_loss = float('inf')
        self.current_stage = "early"
        self.weight_multipliers = self._initialize_weight_multipliers()
    
    def save_scheduler_state(self, filepath: str):
        """保存调度器状态"""
        import json
        
        state = {
            'config': self.config.to_dict(),
            'total_iterations': self.total_iterations,
            'schedule_type': self.schedule_type,
            'loss_history': self.loss_history,
            'adjustment_history': self.adjustment_history,
            'weight_multipliers': self.weight_multipliers,
            'current_stage': self.current_stage,
            'best_loss': self.best_loss,
            'no_improvement_count': self.no_improvement_count
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_scheduler_state(self, filepath: str):
        """加载调度器状态"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.total_iterations = state['total_iterations']
        self.schedule_type = state['schedule_type']
        self.loss_history = state['loss_history']
        self.adjustment_history = state['adjustment_history']
        self.weight_multipliers = state['weight_multipliers']
        self.current_stage = state['current_stage']
        self.best_loss = state['best_loss']
        self.no_improvement_count = state['no_improvement_count']