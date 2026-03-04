"""
GraphSAGE 模型包
"""

# 导入模型以触发注册
from .model import GraphSAGE, MLP
from .party import Server, Client

__all__ = ['GraphSAGE', 'MLP', 'Server', 'Client']
