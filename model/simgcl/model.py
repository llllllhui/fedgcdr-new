"""
SimGCL 模型实现示例 - 简单图对比学习

SimGCL 通过对比学习增强推荐系统的鲁棒性
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


class SimGCLLayer(BaseGNNLayer):
    """
    SimGCL 层 - 带噪声注入的图卷积层
    
    通过在嵌入上添加噪声来创建增强视图
    """
    
    def __init__(self, in_feature: int = None, out_feature: int = None, 
                 eps: float = 0.1):
        super().__init__(in_feature or 16, out_feature or 16)
        self.eps = eps
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                add_noise: bool = False) -> torch.Tensor:
        """
        前向传播 - 带噪声注入的邻居聚合
        
        Args:
            x: 节点特征矩阵
            adj: 邻接矩阵
            add_noise: 是否添加噪声
            
        Returns:
            聚合后的特征矩阵
        """
        # 对称归一化
        degree = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        norm_adj = adj * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)
        
        # 邻居聚合
        output = torch.matmul(norm_adj, x)
        
        # 添加噪声 (创建增强视图)
        if add_noise:
            noise = torch.randn_like(output) * self.eps
            output = output + noise
        
        # L2 归一化
        output = F.normalize(output, p=2, dim=1)
        
        return output


@MODEL_REGISTRY.register('simgcl')
class SimGCL(BaseGNNModel):
    """
    SimGCL 模型 - 简单图对比学习
    
    特点:
    - 对比学习增强鲁棒性
    - 噪声注入提升泛化能力
    - 适合联邦场景下的数据异构性
    
    参考论文: Simple Graph Contrastive Learning for Recommendation
    """
    
    def __init__(self, args, in_feature: int, hid_feature: int = 16,
                 out_feature: int = 16, num_layers: int = 2,
                 eps: float = 0.1, temp: float = 0.2):
        super().__init__(args, in_feature, hid_feature, out_feature)
        
        self.num_layers = num_layers
        self.eps = eps  # 噪声强度
        self.temp = temp  # 对比学习温度参数
        self.device = args.device

        # 构建多层
        self.layers = nn.ModuleList([
            SimGCLLayer(eps=eps).to(self.device) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, is_transfer_stage: bool = False,
                domain_attention: torch.Tensor = None,
                transfer_vec: list = None) -> tuple:
        """
        前向传播
        
        Returns:
            tuple: (x_final, intermediate_embedding, ls, lm)
        """
        intermediate_embedding = []
        ls, lm = 0, 0
        
        # 构建邻接矩阵
        adj = torch.eye(len(x), device=x.device)
        adj[:, 0] = 1.
        adj[0, :] = 1.
        
        # 知识转移阶段
        if is_transfer_stage:
            ls = self.compute_ls(x[0], transfer_vec)
            lm = self.compute_lm(x[0], transfer_vec)
            transfer_vec = torch.stack(transfer_vec)
            x = torch.cat((x, transfer_vec))
            adj = torch.eye(len(x), device=x.device)
            adj[:, 0] = 1.
            adj[0, :] = 1.
        
        # 创建两个增强视图 (用于对比学习)
        layer_outputs_1 = [x]
        layer_outputs_2 = [x]
        
        # 视图 1: 正常传播
        for layer in self.layers:
            x1 = layer(x, adj, add_noise=False)
            layer_outputs_1.append(x1)
            x = x1
        
        # 视图 2: 带噪声传播
        for layer in self.layers:
            x2 = layer(layer_outputs_2[-1], adj, add_noise=True)
            layer_outputs_2.append(x2)
        
        # 平均池化
        x1_final = torch.stack(layer_outputs_1, dim=0).mean(dim=0)
        x2_final = torch.stack(layer_outputs_2, dim=0).mean(dim=0)
        
        # 对比学习损失 (InfoNCE)
        cl_loss = self._contrastive_loss(x1_final[0], x2_final[0], x2_final[1:])
        
        # 记录中间嵌入
        intermediate_embedding.append(layer_outputs_1[0][0].data)
        
        # 返回平均表示
        x_final = (x1_final + x2_final) / 2
        
        # 将对比损失加入到返回的 ls 中 (乘以权重)
        ls = ls + 0.1 * cl_loss
        
        return x_final, intermediate_embedding, ls, lm
    
    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                         negatives: torch.Tensor) -> torch.Tensor:
        """
        计算 InfoNCE 对比损失
        
        Args:
            z1: 视图 1 的用户嵌入
            z2: 视图 2 的用户嵌入 (正样本)
            negatives: 负样本列表
            
        Returns:
            对比损失值
        """
        # 正样本相似度
        pos_score = torch.sum(z1 * z2) / self.temp
        
        # 负样本相似度
        neg_scores = torch.stack([
            torch.sum(z1 * neg) / self.temp for neg in negatives
        ])
        
        # InfoNCE 损失
        loss = -pos_score + torch.logsumexp(
            torch.cat([torch.tensor([pos_score]).to(z1.device), neg_scores]), 
            dim=0
        )
        
        return loss


@MODEL_REGISTRY.register('simgcl_mlp')
class MLP(BaseMLP):
    """
    MLP - 知识向量转换网络 (SimGCL 版本)
    """
    
    def __init__(self, in_feature: int):
        super().__init__(in_feature, hidden_factor=2, dropout=0.1, activation='relu')
