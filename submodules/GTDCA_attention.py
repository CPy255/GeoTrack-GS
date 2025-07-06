import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =================================================================================
# 辅助函数 & 可变形采样核心逻辑
# =================================================================================

def _get_activation_fn(activation):
    """根据字符串返回一个激活函数"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"激活函数应为 relu/gelu, 而不是 {activation}.")


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    多尺度可变形注意力的核心PyTorch实现。
    此函数在'sampling_locations'处从输入的'value'张量中采样特征，
    并使用'attention_weights'将它们组合起来。

    Args:
        value (Tensor): 输入的特征图。形状: (N, S, M, C)
                        N: 批次大小, S: 所有层级的总点数,
                        M: 注意力头的数量, C: 每个头的通道维度。
        value_spatial_shapes (Tensor): 每个特征层级的空间形状。形状: (L, 2)
                                       L: 特征层级的数量。
        sampling_locations (Tensor): 要采样的位置。形状: (N, Lq, M, L, P, 2)
                                     Lq: 查询点的数量, P: 每个查询的采样点数。
        attention_weights (Tensor): 每个采样点的权重。形状: (N, Lq, M, L, P)

    Returns:
        Tensor: 经过注意力计算后的输出特征。形状: (N, Lq, M, C)
    """
    N, S, M, C = value.shape
    _, Lq, M, L, P, _ = sampling_locations.shape

    # 将 'value' 张量按层级切分回不同的特征图
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # grid_sample 函数期望坐标范围在 [-1, 1] 之间, 因此我们从 [0, 1] 进行缩放
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # 为 grid_sample 重塑 value: (N, C, H, W) -> (N*M, C, H, W)
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(N * M, C, H_, W_)

        # 为 grid_sample 重塑采样网格: (N, Lq, M, P, 2) -> (N, Lq*P, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # 使用 grid_sample 执行采样
        # 输出形状为 (N*M, C, Lq, P)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_.unsqueeze(1),
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    # 拼接并重塑采样到的值
    # (N, M, C, Lq, L*P) -> (N, Lq, M, L*P, C)
    attention_weights = attention_weights.transpose(1, 2).reshape(N * M, 1, Lq, L * P)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N, M, C, Lq)

    return output.transpose(1, 2).contiguous()


# =================================================================================
# GT-DCA 模块定义
# =================================================================================

class GTDCALayer(nn.Module):
    """
    单层的几何引导可变形交叉注意力。
    """

    def __init__(self, d_model=256, n_heads=8, n_levels=1, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        # --- 1. 引导机制 (Guidance Mechanism) ---
        # 一个小型的交叉注意力模块，让查询(query)“关注”轨迹点(trajectory points)
        # 这有助于查询理解哪些几何点与其最相关。
        self.guidance_cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.guidance_norm = nn.LayerNorm(d_model)
        # 将轨迹点的2D坐标投射到特征维度(d_model)
        self.trajectory_projection = nn.Linear(2, d_model)

        # --- 2. 可变形采样机制 (Deformable Sampling Mechanism) ---
        # 用于预测采样偏移量和注意力权重的线性层
        # 这些层的输入将是经过几何引导增强后的查询特征
        self.sampling_offset_prediction = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        # --- 3. 输出投影 (Output Projection) ---
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, value, reference_points, trajectory_points):
        """
        Args:
            query (Tensor): 查询特征 (来自高斯函数)。形状: (B, N_g, C)
            value (Tensor): 图像特征图。形状: (B, C, H, W)
            reference_points (Tensor): 查询的初始2D参考点。形状: (B, N_g, 2)
            trajectory_points (Tensor): 用于引导的几何轨迹点。形状: (B, N_t, 2)

        Returns:
            Tensor: 更新后的查询特征。形状: (B, N_g, C)
        """
        B, N_g, C = query.shape
        _, _, H, W = value.shape
        device = query.device

        # --- 步骤 1: 几何引导 ---
        # 将轨迹点坐标投射到特征空间
        # (B, N_t, 2) -> (B, N_t, C)
        trajectory_features = self.trajectory_projection(trajectory_points)

        # 查询(Query)关注轨迹点(trajectory points)以获得引导上下文
        # Q: query, K: trajectory_features, V: trajectory_features
        guidance_context, _ = self.guidance_cross_attn(query, trajectory_features, trajectory_features)

        # 通过添加上下文来创建“被引导的查询”
        # 使用残差连接和层归一化
        guided_query = self.guidance_norm(query + guidance_context)

        # --- 步骤 2: 从“被引导的查询”中预测采样偏移量和权重 ---
        # 预测相对于参考点的偏移量
        # (B, N_g, C) -> (B, N_g, M*L*P*2)
        sampling_offsets = self.sampling_offset_prediction(guided_query)
        sampling_offsets = sampling_offsets.view(B, N_g, self.n_heads, self.n_levels, self.n_points, 2)

        # 预测每个采样点的注意力权重
        # (B, N_g, C) -> (B, N_g, M*L*P)
        attention_weights = self.attention_weights(guided_query).view(B, N_g, self.n_heads, -1)
        attention_weights = F.softmax(attention_weights, -1).view(B, N_g, self.n_heads, self.n_levels, self.n_points)

        # --- 步骤 3: 执行可变形采样 ---
        # 计算最终的采样位置
        # reference_points: (B, N_g, 2) -> (B, N_g, 1, 1, 1, 2)
        # sampling_offsets: (B, N_g, M, L, P, 2)
        # sampling_locations: (B, N_g, M, L, P, 2)
        sampling_locations = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4) + sampling_offsets

        # 为核心函数重塑图像特征图 `value`
        # (B, C, H, W) -> (B, H*W, M, C/M)
        value_reshaped = value.flatten(2).transpose(1, 2)
        value_reshaped = value_reshaped.view(B, H * W, self.n_heads, C // self.n_heads)

        # 定义输入特征图的空间形状
        value_spatial_shapes = torch.as_tensor([[H, W]], dtype=torch.long, device=device)

        # 调用核心的可变形注意力函数
        # 输出形状: (B, N_g, M, C/M)
        sampled_features = ms_deform_attn_core_pytorch(
            value_reshaped, value_spatial_shapes, sampling_locations, attention_weights
        )

        # 重塑回 (B, N_g, C)
        sampled_features = sampled_features.view(B, N_g, C)

        # --- 步骤 4: 最终的输出投影 ---
        output = self.output_proj(sampled_features)

        return output


class GTDCA(nn.Module):
    """ 单个GTDCA层的包装器。可以扩展为多层结构。 """

    def __init__(self, d_model=256, n_heads=8, n_levels=1, n_points=4):
        super().__init__()
        self.layer = GTDCALayer(d_model, n_heads, n_levels, n_points)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, value, reference_points, trajectory_points):
        # 应用GTDCA层
        output = self.layer(query, value, reference_points, trajectory_points)
        # 添加残差连接并进行归一化
        return self.norm(query + output)


# =================================================================================
# 测试脚本
# =================================================================================
if __name__ == '__main__':
    print("--- 测试 GT-DCA 模块 ---")

    # 1. 定义模型参数
    embed_dim = 256
    num_heads = 8
    num_sampling_points = 4

    gtdca_module = GTDCA(
        d_model=embed_dim,
        n_heads=num_heads,
        n_levels=1,  # 为简单起见，假设是单尺度特征
        n_points=num_sampling_points
    )
    print("模块初始化成功。")

    # 2. 生成具有真实形状的模拟数据
    #    批次大小 B=2, 高斯函数数量 N_g=1000, 轨迹点数量 N_t=500
    #    图像特征: H=64, W=80
    B, N_g, N_t, H, W = 2, 1000, 500, 64, 80

    # 查询: 每个高斯函数的特征
    # (B, N_g, C)
    query_feat = torch.rand(B, N_g, embed_dim)

    # 来自CNN骨干网络的输入特征
    # (B, C, H, W)
    src_features = torch.rand(B, embed_dim, H, W)

    # 每个高斯函数在图像平面上的2D参考点
    # 这些点应归一化到 [0, 1]
    # (B, N_g, 2)，最后一个维度是 (x, y)
    reference_points = torch.rand(B, N_g, 2)

    # 2D轨迹点，同样已归一化
    # 这些是关键的几何引导
    # (B, N_t, 2)
    trajectory_points = torch.rand(B, N_t, 2)

    print(f"已生成模拟数据，形状如下:")
    print(f"  查询 (Query): {query_feat.shape}")
    print(f"  图像特征 (Image Features): {src_features.shape}")
    print(f"  参考点 (Reference Points): {reference_points.shape}")
    print(f"  轨迹点 (Trajectory Points): {trajectory_points.shape}")

    # 3. 运行前向传播
    print("\n正在运行 GT-DCA 模块的前向传播...")
    output_features = gtdca_module(
        query=query_feat,
        value=src_features,
        reference_points=reference_points,
        trajectory_points=trajectory_points
    )

    # 4. 检查输出形状
    # 输出应与输入查询的形状相同
    print(f"\n输入查询形状: {query_feat.shape}")
    print(f"输出特征形状: {output_features.shape}")
    assert query_feat.shape == output_features.shape, "形状不匹配！"
    print("\n测试成功！输出形状正确。")
