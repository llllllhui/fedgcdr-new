"""
GraphSAGE 模型实现示例

这是一个模板，展示如何实现新的 GNN 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 导入基类和注册表
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import BaseGNNLayer, BaseGNNModel, BaseMLP
from registry import MODEL_REGISTRY


class GraphSAGELayer(BaseGNNLayer):
    """
    GraphSAGE 层 - 使用 Mean 聚合器
    
    GraphSAGE 通过采样和聚合邻居节点来生成节点嵌入
    """
    
    def __init__(self, in_feature: int, out_feature: int):
        super().__init__(in_feature, out_feature)
        
        # Mean 聚合器的线性变换
        self.linear = nn.Linear(in_feature, out_feature)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - Mean 聚合
        
        Args:
            x: 节点特征矩阵 (num_nodes, embedding_dim)
            adj: 邻接矩阵 (num_nodes, num_nodes)
            
        Returns:
            聚合后的特征矩阵
        """
        # 归一化邻接矩阵
        degree = torch.sum(adj, dim=1, keepdim=True)
        degree_inv = 1.0 / (degree + 1e-8)
        norm_adj = adj * degree_inv
        
        # 邻居聚合 (mean aggregator)
        aggregated = torch.matmul(norm_adj, x)
        
        # 线性变换 + ReLU
        output = F.relu(self.linear(aggregated))
        
        return output


@MODEL_REGISTRY.register('graphsage')
class GraphSAGE(BaseGNNModel):
    """
    GraphSAGE 模型 - 图采样网络
    
    特点:
    - 支持归纳学习
    - 采样邻居聚合，计算效率高
    - 适合动态图结构
    """
    
    def __init__(self, args, in_feature: int, hid_feature: int = 16, 
                 out_feature: int = 16, num_layers: int = 2, 
                 dropout: float = 0.5, aggregator_type: str = 'mean'):
        super().__init__(args, in_feature, hid_feature, out_feature)
        
        self.num_layers = num_layers
        self.aggregator_type = aggregator_type
        
        # 构建多层 GraphSAGE
        self.layers = nn.ModuleList()
        self.device = args.device

        # 输入层到隐藏层
        self.layers.append(GraphSAGELayer(in_feature, hid_feature).to(self.device))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hid_feature, hid_feature).to(self.device))

        # 隐藏层到输出层
        self.layers.append(GraphSAGELayer(hid_feature, out_feature).to(self.device))
        
        self.dropout = nn.Dropout(dropout)
    
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
        intermediate_embedding = []
        ls, lm = 0, 0
        
        # 构建邻接矩阵 (用户与所有物品相连)
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.
        
        # 知识转移阶段
        if is_transfer_stage:
            ls = self.compute_ls(x[0], transfer_vec)
            lm = self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            # 更新邻接矩阵
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        
        # 逐层传播
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            x = self.dropout(x)
            
            # 记录中间嵌入 (第一层后的用户嵌入)
            if i == 0:
                intermediate_embedding.append(x[0].data)
        
        # L2 归一化输出
        x = F.normalize(x, p=2, dim=1)
        
        return x, intermediate_embedding, ls, lm


@MODEL_REGISTRY.register('graphsage_mlp')
class MLP(BaseMLP):
    """
    MLP - 知识向量转换网络 (GraphSAGE 版本)
    """
    
    def __init__(self, in_feature: int):
        super().__init__(in_feature, hidden_factor=2, dropout=0.5, activation='relu')
