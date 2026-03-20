# FedGCDR-New

FedGCDR-New 是一个联邦跨域推荐实验项目，支持多种 GNN 编码器并提供完整的三阶段训练流程：

1. 知识获取（KG）
2. 知识转移（KT）
3. 目标域微调（FT）

训练入口为 `main.py`，模型通过注册表动态加载。

## 环境要求

- Python 3.10+
- 建议使用 CUDA 环境运行（默认设备为 `cuda:0`）

安装依赖：

```bash
pip install -r requirements.txt
```

> `requirements.txt` 仅声明了核心包（`numpy/pandas/tqdm/scikit-learn`），请按你的 CUDA/CPU 环境自行安装 PyTorch。

## 项目结构

```text
fedgcdr-new/
├── main.py                 # 训练入口（参数解析 + KG/KT/FT 流程）
├── utility.py              # 数据集装载（amazon/douban）
├── checkpoint.py           # Checkpoint 保存/加载/恢复
├── Data_Proc.py            # Amazon 数据预处理脚本（4/8/16 域）
├── model/
│   ├── __init__.py         # 自动注册内置模型
│   ├── registry.py         # MODEL/SERVER/CLIENT 注册表
│   ├── base_model.py
│   ├── base_party.py
│   ├── fedgcdr/            # 原始 GAT 相关实现
│   ├── lightgcn/
│   ├── graphsage/
│   ├── simgcl/
│   └── gcn/
├── data/                   # 训练数据
├── checkpoints/            # 阶段 checkpoint（KG/KT）
├── output/                 # 训练日志输出
├── embedding/              # 目标域嵌入产物
└── knowledge_64/           # 知识文件
```

## 快速开始

### 1) 查看可用 checkpoint

```bash
python main.py --list_checkpoints
```

### 2) 直接训练（示例）

```bash
python main.py --gnn_type lightgcn --dataset amazon --num_domain 8 --target_domain 1
```

### 3) 从 KG 阶段恢复

```bash
python main.py --gnn_type lightgcn --resume_from kg --checkpoint_path checkpoints/<kg_checkpoint_dir>
```

### 4) 从 KT 阶段恢复

```bash
python main.py --gnn_type lightgcn --resume_from kt --checkpoint_path checkpoints/<kt_checkpoint_dir>
```

## 可用模型

`main.py --gnn_type` 当前支持：

- `gat`
- `lightgcn`
- `graphsage`
- `simgcl`
- `gcn`

模型、Server、Client 通过 `model/registry.py` 统一管理，并在 `model/__init__.py` 自动注册。

## 关键参数（以 `main.py` 为准）

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--dataset` | 数据集 | `amazon` |
| `--num_domain` | 域数量 | `8` |
| `--target_domain` | 目标域索引 | `1` |
| `--gnn_type` | GNN 类型 | `gat` |
| `--round_gat` | KG/KT 阶段轮数 | `30` |
| `--round_ft` | FT 阶段轮数 | `60` |
| `--embedding_size` | 嵌入维度 | `16` |
| `--device` | 训练设备 | `cuda:0` |
| `--resume_from` | checkpoint 恢复阶段 | `kg` / `kt` |
| `--checkpoint_path` | checkpoint 目录路径 | `None` |
| `--random_seed` | 随机种子 | `42` |

## 数据准备说明

- `amazon`：由 `Data_Proc.py` 生成 `data/{4|8|16}domains` 下的 `implicit.json` 与 `domain_user.json`。
- `douban`：`utility.py` 默认读取 `data/douban_oldver/` 下对应文件。

训练前需确保对应数据文件已就位，否则会在数据加载阶段报错。

## 训练输出

- 日志：`output/*.out`（含参数与关键指标）
- 指标：`hr_5 / ndcg_5 / hr_10 / ndcg_10`
- checkpoint：`checkpoints/kg_*`、`checkpoints/kt_*`（默认最多保留最近 3 份）
- 嵌入：`embedding/<model>/...json`

## 许可证

MIT
