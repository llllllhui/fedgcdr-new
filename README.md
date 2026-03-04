# FedGCDR - 联邦跨域推荐系统

## 项目结构

```
FedGCDR/
├── model/                      # 模型目录
│   ├── __init__.py            # 包初始化，自动注册内置模型
│   ├── registry.py            # 模型注册表
│   ├── base_model.py          # 模型基类接口
│   ├── base_party.py          # Server/Client 基类
│   │
│   └── fedgcdr/               # FedGCDR 模型实现
│       ├── __init__.py
│       ├── model.py           # GAT 模型
│       ├── lightgcn_model.py  # LightGCN 模型
│       ├── party.py           # GAT 的 Server/Client
│       └── party_lightgcn.py  # LightGCN 的 Server/Client
│
├── main.py                     # 主训练脚本
├── utility.py                  # 工具函数
├── checkpoint.py              # Checkpoint 管理
└── requirements.txt           # 依赖
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
# 使用 LightGCN 模型
python main.py --gnn_type lightgcn --target_domain 1

# 使用 GAT 模型
python main.py --gnn_type gat --target_domain 1

# 使用 GraphSAGE 模型 (待实现)
python main.py --gnn_type graphsage --target_domain 1

# 使用 SimGCL 模型 (待实现)
python main.py --gnn_type simgcl --target_domain 1
```

### 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gnn_type` | 模型类型 | lightgcn |
| `--dataset` | 数据集 | amazon |
| `--num_domain` | 域数量 | 4 |
| `--target_domain` | 目标域 | 1 |
| `--embedding_size` | 嵌入维度 | 16 |
| `--round_gat` | GNN 训练轮数 | 30 |
| `--round_ft` | 微调轮数 | 60 |

## 添加新模型

### 1. 创建模型目录

```bash
mkdir model/graphsage
```

### 2. 实现模型类

```python
# model/graphsage/model.py
import torch
import torch.nn as nn
from model.base_model import BaseGNNLayer, BaseGNNModel
from model.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register('graphsage')
class GraphSAGE(BaseGNNModel):
    def __init__(self, args, in_feature, hid_feature=16, out_feature=16):
        super().__init__(args, in_feature, hid_feature, out_feature)
        # 实现你的模型
    
    def forward(self, x, is_transfer_stage=False, 
                domain_attention=None, transfer_vec=None):
        # 实现前向传播
        pass
```

### 3. 实现 Server/Client

```python
# model/graphsage/party.py
from model.base_party import BaseServer, BaseClient
from model.registry import SERVER_REGISTRY, CLIENT_REGISTRY
from .model import GraphSAGE

@SERVER_REGISTRY.register('graphsage')
class Server(BaseServer):
    def __init__(self, id, d_name, num_m, total_clients, clients, 
                 evaluate_data, user_dic, args):
        super().__init__(id, d_name, num_m, total_clients, clients, 
                        evaluate_data, user_dic, args)
        self.gnn_model = GraphSAGE(args, args.embedding_size)
    
    def get_gnn_model(self):
        return self.gnn_model
    
    def test_gnn(self, epoch_id):
        self.gnn_model.eval()
        return self.test(self.user_embedding_with_attention, self.V, epoch_id)
    
    def kt_stage(self, tf_flag=False, **kwargs):
        # 实现知识转移逻辑
        pass

@CLIENT_REGISTRY.register('graphsage')
class Client(BaseClient):
    def __init__(self, id, train_data, num_m, rating_mean, domain_names, args):
        super().__init__(id, train_data, num_m, rating_mean, domain_names, args)
        self.gnn_model = None
    
    def get_gnn_model(self):
        return self.gnn_model
    
    def train_gnn(self, **kwargs):
        # 实现训练逻辑
        pass
```

### 4. 注册 MLP

```python
# model/graphsage/model.py (续)
from model.base_model import BaseMLP
from model.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register('graphsage_mlp')
class MLP(BaseMLP):
    def __init__(self, in_feature):
        super().__init__(in_feature)
```

### 5. 运行训练

```bash
python main.py --gnn_type graphsage --target_domain 1
```

## 查看可用模型

```python
from model.registry import list_all_models
print(list_all_models())
```

## 可用模型

| 模型 | 描述 | 状态 |
|------|------|------|
| `gat` | 图注意力网络 | ✅ 已实现 |
| `lightgcn` | 轻量级图卷积 | ✅ 已实现 |
| `graphsage` | 图采样网络 | 🔄 待实现 |
| `simgcl` | 简单图对比学习 | 🔄 待实现 |

## Checkpoint 管理

```bash
# 列出可用 checkpoint
python main.py --list_checkpoints

# 从知识获取阶段恢复
python main.py --resume_from kg --checkpoint_path checkpoints/kg_xxx.pt

# 从知识转移阶段恢复
python main.py --resume_from kt --checkpoint_path checkpoints/kt_xxx.pt
```

## 许可证

MIT License
