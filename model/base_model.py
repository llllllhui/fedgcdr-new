"""
GNN 模型基类接口 - 定义所有图神经网络模型的统一接口

所有新模型必须继承 BaseGNNLayer 和 BaseGNNModel
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseGNNLayer(ABC, nn.Module):
    """GNN 层基类 - 所有 GNN 层必须继承此类"""
    
    def __init__(self, in_feature: int, out_feature: int, **kwargs):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
    
    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征矩阵 (num_nodes, embedding_dim)
            adj: 邻接矩阵 (num_nodes, num_nodes)
            
        Returns:
            输出特征矩阵
        """
        pass


class BaseGNNModel(ABC, nn.Module):
    """
    GNN 模型基类 - 所有 GNN 模型必须继承此类
    
    定义了 FedGCDR 框架所需的标准接口
    """
    
    def __init__(self, args, in_feature: int, hid_feature: int = 16, 
                 out_feature: int = 16, **kwargs):
        super().__init__()
        self.args = args
        self.in_feature = in_feature
        self.hid_feature = hid_feature
        self.out_feature = out_feature
        self.device = args.device
    
    @abstractmethod
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
                - x_final: 最终输出特征
                - intermediate_embedding: 中间嵌入（用于知识提取）
                - ls: 知识相似度损失
                - lm: MSE 损失
        """
        pass
    
    @staticmethod
    def compute_ls(f_t: torch.Tensor, f_s: list) -> torch.Tensor:
        """
        计算知识相似度损失 (Knowledge Similarity Loss)
        
        Args:
            f_t: 目标域特征
            f_s: 源域特征列表
            
        Returns:
            相似度损失值
        """
        total_sim = 0
        for fs in f_s:
            with torch.no_grad():
                sim = (torch.cosine_similarity(fs, f_t, dim=0) + 1) / 2
                total_sim += sim
        
        F_s = 0
        for fs in f_s:
            with torch.no_grad():
                sim = (torch.cosine_similarity(fs, f_t, dim=0) + 1) / 2
            F_s += sim * fs / total_sim
        
        loss = torch.norm(f_t - F_s) ** 2
        return loss
    
    @staticmethod
    def compute_lm(f_t: torch.Tensor, f_s: list) -> torch.Tensor:
        """
        计算均方误差损失 (Mean Square Error Loss)
        
        Args:
            f_t: 目标域特征
            f_s: 源域特征列表
            
        Returns:
            MSE 损失值
        """
        loss = 0
        for fs in f_s:
            loss += torch.nn.functional.mse_loss(fs, f_t)
        return loss
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.out_feature
    
    def reset_parameters(self):
        """重置模型参数"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


class BaseMLP(nn.Module):
    """
    MLP 基类 - 用于知识向量转换
    
    所有模型的知识转换 MLP 应继承此类
    """
    
    def __init__(self, in_feature: int, hidden_factor: int = 2, 
                 dropout: float = 0.0, activation: str = 'tanh'):
        super().__init__()
        self.in_feature = in_feature
        self.hidden_feature = in_feature * hidden_factor
        self.out_feature = in_feature // 2
        
        # 激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"不支持的激活函数：{activation}")
        
        # 三层 MLP 结构
        self.L1 = nn.Linear(in_feature, self.hidden_feature)
        self.L2 = nn.Linear(self.hidden_feature, self.out_feature)
        self.L3 = nn.Linear(self.out_feature, in_feature)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.activation(self.L1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(self.L2(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(self.L3(x))
        return x
    
    def reset_parameters(self):
        """重置参数"""
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.zeros_(self.L1.bias)
        nn.init.zeros_(self.L2.bias)
        nn.init.zeros_(self.L3.bias)
