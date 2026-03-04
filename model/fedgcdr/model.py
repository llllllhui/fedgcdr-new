"""
GAT 模型实现 - 图注意力网络

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


class GATLayer(BaseGNNLayer):
    """
    GAT 层 - 图注意力网络层
    
    使用可学习的注意力向量 A 计算节点间的注意力系数
    """
    
    def __init__(self, in_feature: int, out_feature: int, alpha: float = 0.1):
        super().__init__(in_feature, out_feature)
        # 注意力矩阵 A, 形状：(2 * out_feature, 1)
        self.A = nn.Parameter(torch.empty(size=(2 * out_feature, 1)))
        nn.init.xavier_uniform_(self.A.data, nn.init.calculate_gain('relu'))
        self.alpha = alpha
    
    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input: 输入特征矩阵
            adj: 邻接矩阵
            
        Returns:
            输出特征矩阵
        """
        h = input
        h1 = torch.matmul(h, self.A[self.out_feature:, :])
        h2 = torch.matmul(h, self.A[:self.out_feature, :])
        e = h1 + h2.T
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = nn.functional.softmax(attention, dim=-1)
        ah = torch.matmul(attention, h)
        return ah


@MODEL_REGISTRY.register('gat')
class GAT(BaseGNNModel):
    """
    GAT 模型 - 图注意力网络
    
    两层 GAT 结构，用于学习用户 - 物品图的嵌入表示
    """
    
    def __init__(self, args, in_feature: int, hid_feature: int = 16, 
                 out_feature: int = 16, alpha: float = 0.1, dropout: float = 0):
        super().__init__(args, in_feature, hid_feature, out_feature)
        self.drop = nn.Dropout(p=dropout)
        self.in2hidden = GATLayer(in_feature, hid_feature, alpha).to(args.device)
        self.hidden2out = GATLayer(hid_feature, out_feature, alpha).to(args.device)
    
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
            tuple: (x, intermediate_embedding, ls, lm)
        """
        ls, lm = 0, 0
        alpha, beta = 0.01, 0.01
        intermediate_embedding = []
        
        # 构建邻接矩阵
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.
        
        # 第一层
        x = self.in2hidden(x, adj)
        intermediate_embedding.append(x[0].data)
        
        # 知识转移阶段
        if is_transfer_stage:
            ls = alpha / 2 * self.compute_ls(x[0], transfer_vec)
            lm = beta / 2 * self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        
        # 第二层
        x = self.hidden2out(x, adj)
        return x, intermediate_embedding, ls, lm


@MODEL_REGISTRY.register('gat_mlp')
class MLP(BaseMLP):
    """
    MLP - 知识向量转换网络
    
    将源域知识转换为目标域知识
    """
    
    def __init__(self, in_feature: int):
        super().__init__(in_feature, hidden_factor=2, dropout=0.0, activation='tanh')
