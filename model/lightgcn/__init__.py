"""
LightGCN 模型包
"""

# 导入模型以触发注册
from .model import LightGCN, MLP
from .party import Server, Client

__all__ = ['LightGCN', 'MLP', 'Server', 'Client']
