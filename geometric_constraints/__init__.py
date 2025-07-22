"""
高级几何约束和轨迹跟踪模块

该模块提供了用于 GeoTrack-GS 的高级几何约束功能，包括：
- 自适应重投影约束系统
- 多尺度几何一致性约束
- 智能轨迹质量评估和管理
- 动态几何约束权重调度
- 实时几何约束验证
"""

from .data_structures import Trajectory, Point2D, ConstraintResult
from .interfaces import ConstraintEngine, QualityAssessor, TrajectoryManager
from .config import ConstraintConfig
from .trajectory_manager import TrajectoryManagerImpl, QualityAssessorImpl
from .multiscale_constraints import (
    MultiScaleImagePyramid, 
    MultiScaleConstraints, 
    MultiScaleConstraintEngine,
    ScaleConsistencyConstraint,
    EnhancedMultiScaleConstraintEngine
)
from .constraint_engine import (
    EnhancedReprojectionConstraint,
    ConstraintFusion,
    GeometricConsistencyChecker,
    ConstraintEngineImpl,
    AdvancedOutlierDetector,
    EnhancedConstraintEngine,
    HuberLoss,
    RobustEstimator,
    ReprojectionStats
)
from .reprojection_validator import (
    ValidationMetrics,
    QualityReport,
    GeometricConstraintSatisfactionCalculator,
    ReprojectionValidator,
    ConstraintParameterCalibrator,
    EnhancedReprojectionValidator,
    create_reprojection_validator,
    analyze_validation_trends
)

__all__ = [
    'Trajectory',
    'Point2D', 
    'ConstraintResult',
    'ConstraintEngine',
    'QualityAssessor',
    'TrajectoryManager',
    'TrajectoryManagerImpl',
    'QualityAssessorImpl',
    'ConstraintConfig',
    'MultiScaleImagePyramid',
    'MultiScaleConstraints',
    'MultiScaleConstraintEngine',
    'ScaleConsistencyConstraint',
    'EnhancedMultiScaleConstraintEngine',
    'EnhancedReprojectionConstraint',
    'ConstraintFusion',
    'GeometricConsistencyChecker',
    'ConstraintEngineImpl',
    'AdvancedOutlierDetector',
    'EnhancedConstraintEngine',
    'HuberLoss',
    'RobustEstimator',
    'ReprojectionStats',
    'ValidationMetrics',
    'QualityReport',
    'GeometricConstraintSatisfactionCalculator',
    'ReprojectionValidator',
    'ConstraintParameterCalibrator',
    'EnhancedReprojectionValidator',
    'create_reprojection_validator',
    'analyze_validation_trends'
]