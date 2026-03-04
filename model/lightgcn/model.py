"""
LightGCN 模型实现 - 轻量级图卷积网络

用于 FedGCDR 联邦跨域推荐系统
"""

import torch.nn as nn
import torch
import sys
import os

# 导入基类和注册表
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseGNNLayer, BaseGNNModel, BaseMLP
from registry import MODEL_REGISTRY


class LightGCNLayer(BaseGNNLayer):
    """
    LightGCN 层 - 轻量级图卷积层

    去除了特征变换和非线性激活，仅保留邻接聚合
    """

    def __init__(self, in_feature: int = None, out_feature: int = None):
        # LightGCN 不进行特征变换，in/out_feature 仅用于接口兼容
        super().__init__(in_feature or 16, out_feature or 16)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 简单的邻居聚合

        Args:
            x: 节点特征矩阵 (num_nodes, embedding_dim)
            adj: 邻接矩阵 (num_nodes, num_nodes)

        Returns:
            聚合后的特征矩阵
        """
        # 对称归一化：D^(-1/2) * A * D^(-1/2)
        degree = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        norm_adj = adj * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)

        # 邻居聚合
        output = torch.matmul(norm_adj, x)
        return output


@MODEL_REGISTRY.register('lightgcn')
class LightGCN(BaseGNNModel):
    """
    LightGCN 模型 - 轻量级图卷积网络

    使用多层传播和层归一化，专注于图结构信息的传播
    """

    def __init__(self, args, in_feature: int, hid_feature: int = 16,
                 out_feature: int = 16, num_layers: int = 1, dropout: float = 0.1):
        super().__init__(args, in_feature, hid_feature, out_feature)
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout)
        self.device = args.device

        # LightGCN 使用多个层，每层都是简单的邻接聚合
        self.layers = nn.ModuleList([LightGCNLayer().to(self.device) for _ in range(num_layers)])

        # 残差连接投影层
        self.res_proj = nn.Linear(hid_feature, hid_feature).to(self.device) if num_layers > 1 else None

    def forward(self, x: torch.Tensor, is_transfer_stage: bool = False,
                domain_attention: torch.Tensor = None,
                transfer_vec: list = None) -> tuple:
        """
        前向传播

        Args:
            x: 输入特征矩阵，第一行是用户嵌入，其他是物品嵌入
            is_transfer_stage: 是否为知识转移阶段
            domain_attention: 域注意力向量
            transfer_vec: 待转移的知识向量列表

        Returns:
            tuple: (x_final, intermediate_embedding, ls, lm)
        """
        ls, lm = 0, 0
        alpha, beta = 0.01, 0.01
        intermediate_embedding = []

        # 构建邻接矩阵
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.

        # 存储每一层的输出
        layer_outputs = []

        # 知识转移阶段预处理
        if is_transfer_stage:
            ls = alpha / 2 * self.compute_ls(x[0], transfer_vec)
            lm = beta / 2 * self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            # 更新邻接矩阵
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        else:
            # 非转移阶段：包含原始输入
            layer_outputs.append(x)

        # LightGCN 的逐层传播
        for i, layer in enumerate(self.layers):
            x_new = layer(x, adj)
            # LayerNorm
            ln = nn.LayerNorm(self.hid_feature, device=x.device)
            x_new = ln(x_new)
            # Dropout
            x_new = self.drop(x_new)
            # 残差连接
            if self.res_proj is not None and i > 0:
                x_new = x_new + self.res_proj(x)
            x = x_new
            layer_outputs.append(x)

        # 所有层的平均作为最终表示
        x_final = torch.stack(layer_outputs, dim=0).mean(dim=0)

        # 提取中间嵌入
        intermediate_embedding.append(layer_outputs[0][0].data)

        return x_final, intermediate_embedding, ls, lm


@MODEL_REGISTRY.register('lightgcn_mlp')
class MLP(BaseMLP):
    """
    MLP - 知识向量转换网络 (LightGCN 版本)

    将源域知识转换为目标域知识
    """

    def __init__(self, in_feature: int):
        super().__init__(in_feature, hidden_factor=2, dropout=0.0, activation='tanh')
