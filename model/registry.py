"""
模型注册表 - 用于动态注册和管理不同的 GNN 模型

使用示例:
    from model.registry import MODEL_REGISTRY
    
    # 注册新模型
    @MODEL_REGISTRY.register('graphsage')
    class GraphSAGEModel(BaseGNNModel):
        pass
    
    # 获取模型
    model_cls = MODEL_REGISTRY.get('graphsage')
"""


class Registry:
    """通用注册表基类"""
    
    def __init__(self, name="registry"):
        self._name = name
        self._registry_map = {}
    
    def register(self, name=None):
        """注册装饰器"""
        def wrapper(cls):
            key = name if name is not None else cls.__name__
            if key in self._registry_map:
                raise ValueError(f"名称 '{key}' 已在 {self._name} 中注册")
            self._registry_map[key] = cls
            return cls
        return wrapper
    
    def get(self, name):
        """获取注册的类"""
        if name not in self._registry_map:
            available = list(self._registry_map.keys())
            raise KeyError(f"'{name}' 未在 {self._name} 中注册。可用选项：{available}")
        return self._registry_map[name]
    
    def list_available(self):
        """列出所有已注册的名称"""
        return list(self._registry_map.keys())
    
    def __contains__(self, name):
        return name in self._registry_map


# 全局模型注册表
MODEL_REGISTRY = Registry("model")

# 全局 Server/Client 注册表
SERVER_REGISTRY = Registry("server")
CLIENT_REGISTRY = Registry("client")


def get_model_class(model_name):
    """根据模型名称获取模型类"""
    return MODEL_REGISTRY.get(model_name)


def get_server_class(model_name):
    """根据模型名称获取 Server 类"""
    return SERVER_REGISTRY.get(model_name)


def get_client_class(model_name):
    """根据模型名称获取 Client 类"""
    return CLIENT_REGISTRY.get(model_name)


def list_all_models():
    """列出所有可用的模型"""
    return {
        'models': MODEL_REGISTRY.list_available(),
        'servers': SERVER_REGISTRY.list_available(),
        'clients': CLIENT_REGISTRY.list_available(),
    }
