"""
SimGCL 模型包
"""

# 导入模型以触发注册
from .model import SimGCL, MLP
from .party import Server, Client

__all__ = ['SimGCL', 'MLP', 'Server', 'Client']
