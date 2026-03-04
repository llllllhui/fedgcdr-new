"""
FedGCDR 模型包

模型目录结构:
    model/
    ├── __init__.py          # 包初始化
    ├── registry.py          # 模型注册表
    ├── base_model.py        # 模型基类接口
    ├── base_party.py        # Server/Client 基类
    │
    ├── fedgcdr/             # FedGCDR 模型 (GAT + LightGCN)
    │   ├── __init__.py
    │   ├── model.py         # GAT 模型实现
    │   ├── lightgcn_model.py # LightGCN 模型实现
    │   ├── party.py         # GAT 的 Server/Client
    │   └── party_lightgcn.py # LightGCN 的 Server/Client
    │
    ├── graphsage/           # GraphSAGE 模型 (示例)
    │   ├── __init__.py
    │   ├── model.py         # GraphSAGE 模型实现
    │   └── party.py         # GraphSAGE 的 Server/Client
    │
    └── simgcl/              # SimGCL 模型 (示例)
        ├── __init__.py
        ├── model.py         # SimGCL 模型实现
        └── party.py         # SimGCL 的 Server/Client
"""

from .registry import (
    MODEL_REGISTRY,
    SERVER_REGISTRY,
    CLIENT_REGISTRY,
    get_model_class,
    get_server_class,
    get_client_class,
    list_all_models,
)

from .base_model import BaseGNNLayer, BaseGNNModel, BaseMLP

__all__ = [
    # 注册表
    'MODEL_REGISTRY',
    'SERVER_REGISTRY', 
    'CLIENT_REGISTRY',
    'get_model_class',
    'get_server_class',
    'get_client_class',
    'list_all_models',
    # 基类
    'BaseGNNLayer',
    'BaseGNNModel',
    'BaseMLP',
]

# 自动注册内置模型
def _register_builtin_models():
    """注册内置模型"""
    # 注册 GAT 模型
    try:
        from .fedgcdr.model import GAT, MLP as GATMLP
        from .fedgcdr.party import Server as GATServer, Client as GATClient

        MODEL_REGISTRY.register('gat')(GAT)
        MODEL_REGISTRY.register('gat_mlp')(GATMLP)
        SERVER_REGISTRY.register('gat')(GATServer)
        CLIENT_REGISTRY.register('gat')(GATClient)
    except ImportError:
        pass

    # 注册 LightGCN 模型
    try:
        from .lightgcn.model import LightGCN, MLP as LightGCNMLP
        from .lightgcn.party import Server as LightGCNServer, Client as LightGCNClient

        MODEL_REGISTRY.register('lightgcn')(LightGCN)
        MODEL_REGISTRY.register('lightgcn_mlp')(LightGCNMLP)
        SERVER_REGISTRY.register('lightgcn')(LightGCNServer)
        CLIENT_REGISTRY.register('lightgcn')(LightGCNClient)
    except ImportError:
        pass

    # 注册 GraphSAGE 模型
    try:
        from .graphsage.model import GraphSAGE, MLP as GraphSAGEMLP
        from .graphsage.party import Server as GraphSAGEServer, Client as GraphSAGEClient

        MODEL_REGISTRY.register('graphsage')(GraphSAGE)
        MODEL_REGISTRY.register('graphsage_mlp')(GraphSAGEMLP)
        SERVER_REGISTRY.register('graphsage')(GraphSAGEServer)
        CLIENT_REGISTRY.register('graphsage')(GraphSAGEClient)
    except ImportError:
        pass

    # 注册 SimGCL 模型
    try:
        from .simgcl.model import SimGCL, MLP as SimGCLMLP
        from .simgcl.party import Server as SimGCLServer, Client as SimGCLClient

        MODEL_REGISTRY.register('simgcl')(SimGCL)
        MODEL_REGISTRY.register('simgcl_mlp')(SimGCLMLP)
        SERVER_REGISTRY.register('simgcl')(SimGCLServer)
        CLIENT_REGISTRY.register('simgcl')(SimGCLClient)
    except ImportError:
        pass


_register_builtin_models()
