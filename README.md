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

> 请按你的 CUDA/CPU 环境自行安装 PyTorch（推荐 2.0+）。

## 项目结构

```text
fedgcdr-new/
├── main.py                 # 训练入口（参数解析 + KG/KT/FT 流程）
├── utility.py              # 数据集装载（amazon/douban）
├── checkpoint.py           # Checkpoint 保存/加载/恢复
├── Data_Proc.py            # Amazon 数据预处理脚本（支持 2/4/8/16 域）
├── domain_config.py        # 域名称与核心阈值中心化配置
├── training_targets.py     # 多目标域训练辅助函数
│
├── backend/                # FastAPI 管理平台后端
│   ├── main.py             # FastAPI 应用入口
│   ├── requirements.txt    # 后端 Python 依赖
│   ├── core/config.py      # Pydantic-Settings 配置
│   ├── db/                 # 数据库（SQLAlchemy ORM + SQLite）
│   ├── auth/               # JWT 认证 + 密码哈希
│   ├── schemas/            # Pydantic 请求/响应 Schema
│   └── api/                # REST + WebSocket 路由
│       ├── auth.py         # /api/auth/ 认证
│       ├── training.py     # /api/training/ 训练任务 CRUD
│       ├── checkpoint.py   # /api/checkpoints/ 管理
│       ├── recommendation.py # /api/recommendations/ 推荐查询
│       ├── ws.py           # /api/ws/ WebSocket 实时推送
│       └── ws_manager.py   # WebSocket 连接管理
│
├── frontend/               # React + Vite 管理平台前端
│   ├── src/
│   │   ├── App.tsx         # 路由 + 布局
│   │   ├── api/client.ts   # Axios HTTP 客户端
│   │   ├── hooks/          # WebSocket 实时 Hook
│   │   └── pages/          # 看板 / 训练管理 / 推荐查询 / 登录
│   ├── package.json
│   └── vite.config.ts
│
├── model/
│   ├── __init__.py         # 自动注册内置模型
│   ├── registry.py         # MODEL/SERVER/CLIENT 注册表
│   ├── base_model.py       # GNN 模型基类
│   ├── base_party.py       # 联邦通信基类
│   ├── fedgcdr/            # 原始 GAT 实现
│   ├── lightgcn/           # LightGCN 模型
│   ├── graphsage/          # GraphSAGE 模型
│   └── gcn/                # GCN 模型
│
├── data/                   # 训练数据（2/4/8/16 域 Amazon）
├── checkpoints/            # 阶段 checkpoint（KG/KT）
├── output/                 # 训练日志输出
├── embedding/              # 目标域嵌入产物
├── knowledge_64/           # 知识迁移文件
└── training-results-web/   # 旧版静态前端 + 推荐数据构建脚本
```

## Web 管理平台

项目附带一套完整的 **前后端分离管理平台**，支持训练任务远程创建、实时监控 WebSocket 推送、Checkpoint 管理、推荐结果查询。

### 架构概览

```
浏览器 (Vite+React)  ←HTTP/WS→  FastAPI 后端  → 子进程 main.py 训练
                                  │
                                  ↓
                              SQLite (fedgcdr.db)
```

### 启动后端

```bash
# 1. 安装后端依赖
pip install -r backend/requirements.txt

# 2. 启动 FastAPI 服务（项目根目录）
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8080
```

启动后输出：
```
[OK] Database ready (sqlite:///./fedgcdr.db)
[OK] FedGCDR System - FedGCDR 毕设系统
```

服务监听 `http://127.0.0.1:8080`，自动创建 SQLite 数据库并注册所有 API 路由。

### 启动前端

```bash
# 1. 安装前端依赖
cd frontend && npm install

# 2. 启动 Vite 开发服务器
npm run dev
```

浏览器打开 `http://localhost:5173/`。  
Vite 自动将 `/api/*` 和 WebSocket 请求代理到后端 8080 端口。

### 首次使用流程

1. 启动后端 + 前端后，在浏览器注册账号 → 登录
2. 进入「训练管理」页，选择 GNN 模型、域数等参数，创建训练任务
3. 训练过程中实时查看指标曲线和日志推送
4. 训练完成后，生成推荐快照数据：

   ```bash
   python training-results-web/scripts/build_recommendation_data.py
   ```

5. 进入「推荐查询」页，选择快照、输入用户索引，查看跨域前/后的 Top10 推荐对比

## 快速开始

### 1) 查看可用 checkpoint

```bash
python main.py --list_checkpoints
```

### 2) 直接训练（示例）

```bash
python main.py --gnn_type lightgcn --dataset amazon --num_domain 8 --target_domain 1
```

双域训练示例：

```bash
python main.py --gnn_type gat --dataset amazon --num_domain 2 --target_domain -1
```

上面的命令会依次运行：

- `Books -> CDs`
- `CDs -> Books`

### 3) 从 KG 阶段恢复

```bash
python main.py --gnn_type lightgcn --resume_from kg --checkpoint_path checkpoints/<kg_checkpoint_dir>
```

### 4) 从 KT 阶段恢复

```bash
python main.py --gnn_type lightgcn --resume_from kt --checkpoint_path checkpoints/<kt_checkpoint_dir>
```

### 5) 列出所有可用 checkpoint

```bash
python main.py --list_checkpoints
```

### 6) 实时训练监控

启用实时指标追踪（默认启用）：

- 实时图表：`output/figures/live/*_live_metrics.png`
- 实时数据：`output/figures/live/*_live_metrics.csv`

可以通过以下参数控制：

```bash
python main.py --gnn_type lightgcn --live_plot True --live_plot_refresh_every 5
```

## 可用模型

`main.py --gnn_type` 当前支持：

- `gat`
- `lightgcn`
- `graphsage`
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
| `--resume_from` | checkpoint 恢复阶段 (kg/kt/None) | `kg` |
| `--checkpoint_path` | checkpoint 目录路径 | 见代码注释 |
| `--random_seed` | 随机种子 | `42` |
| `--live_plot` | 启用实时指标图表 | `True` |
| `--live_plot_dir` | 实时图表输出目录 | `output/figures/live` |
| `--live_plot_refresh_every` | 每N轮刷新图表 | `1` |
| `--lr_gnn` | GNN 模型统一学习率 | `0.001` |
| `--weight_decay` | 权重衰减 | `1e-4` |
| `--local_epoch` | 本地训练轮数 | `3` |
| `--user_batch` | 用户批次大小 | `16` |

## 数据准备说明

- `amazon`：由 `Data_Proc.py` 生成 `data/{2|4|8|16}domains` 下的 `implicit.json`、`domain_user.json` 和 `domain_names.json`。
- `douban`：`utility.py` 默认读取 `data/douban_oldver/` 下对应文件。

双域 Amazon 数据可直接生成，例如：

```bash
python Data_Proc.py --num_domains 2
```

如果不传 `--domains`，会使用默认双域预设 `Books + CDs`。

训练前需确保对应数据文件已就位，否则会在数据加载阶段报错。

## 训练输出

- 日志：`output/*.out`（含参数与关键指标）
- 指标：`hr_5 / ndcg_5 / hr_10 / ndcg_10`
- checkpoint：`checkpoints/kg_*`、`checkpoints/kt_*`（默认最多保留最近 3 份）
- 嵌入：`embedding/<model>/...json`

## 许可证

MIT
