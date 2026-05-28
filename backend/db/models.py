"""ORM 数据模型"""

import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, JSON, Enum as SAEnum,
)
from sqlalchemy.orm import relationship
import enum

from backend.db.database import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    VIEWER = "viewer"


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ─────────── User ───────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(16), default=UserRole.VIEWER.value, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    training_tasks = relationship("TrainingTask", back_populates="creator")

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# ─────────── Training Task ───────────

class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id = Column(Integer, primary_key=True, index=True)
    # 任务元信息
    name = Column(String(128), nullable=True, comment="任务名称")
    description = Column(Text, nullable=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # 训练参数（快照用户提交的配置）
    gnn_type = Column(String(32), nullable=False)  # gat / lightgcn / graphsage / gcn
    dataset = Column(String(32), default="amazon")
    num_domain = Column(Integer, default=4)
    target_domain = Column(Integer, default=1)
    embedding_size = Column(Integer, default=16)
    round_gat = Column(Integer, default=30)
    round_ft = Column(Integer, default=60)
    lr_gnn = Column(Float, default=0.001)
    lr_mf = Column(Float, default=0.005)
    dp = Column(Boolean, default=True)
    eps = Column(Float, default=8.0)
    random_seed = Column(Integer, default=42)
    local_epoch = Column(Integer, default=3)
    user_batch = Column(Integer, default=16)
    weight_decay = Column(Float, default=1e-4)
    num_negative = Column(Integer, default=4)
    knowledge_gate = Column(Boolean, default=True)
    knowledge_gate_threshold = Column(Float, default=0.5)
    extra_params = Column(JSON, nullable=True)

    # 状态
    status = Column(String(20), default=TrainingStatus.PENDING.value)
    progress = Column(Float, default=0.0, comment="0-100 进度百分比")
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # 训练输出
    output_file = Column(String(256), nullable=True, comment="main.py 的 stdout 日志路径")
    checkpoint_paths = Column(JSON, nullable=True, comment="生成的 checkpoint 目录列表")
    pid = Column(Integer, nullable=True, comment="子进程 PID")

    # 最佳指标摘要
    best_hr5 = Column(Float, nullable=True)
    best_hr10 = Column(Float, nullable=True)
    best_ndcg5 = Column(Float, nullable=True)
    best_ndcg10 = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    creator = relationship("User", back_populates="training_tasks")
    metrics = relationship("MetricRecord", back_populates="task", cascade="all, delete-orphan")
    logs = relationship("TrainingLog", back_populates="task", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TrainingTask #{self.id} {self.gnn_type} {self.num_domain}domains [{self.status}]>"


# ─────────── Metric Record ───────────

class MetricRecord(Base):
    """每轮的训练指标记录"""
    __tablename__ = "metric_records"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("training_tasks.id", ondelete="CASCADE"), nullable=False)
    step = Column(Integer, nullable=False, comment="全局步数")
    stage = Column(String(8), nullable=False, comment="KG / KT / FT")
    domain = Column(String(64), nullable=False, comment="域名称")
    round = Column(Integer, nullable=False, comment="该阶段的轮次")
    hr_5 = Column(Float, nullable=True)
    ndcg_5 = Column(Float, nullable=True)
    hr_10 = Column(Float, nullable=True)
    ndcg_10 = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    task = relationship("TrainingTask", back_populates="metrics")


# ─────────── Training Log ───────────

class TrainingLog(Base):
    """训练日志行（用于实时终端展示）"""
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("training_tasks.id", ondelete="CASCADE"), nullable=False)
    level = Column(String(8), default="INFO")  # INFO / WARN / ERROR
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    task = relationship("TrainingTask", back_populates="logs")


# ─────────── Checkpoint Metadata ───────────

class Checkpoint(Base):
    """Checkpoint 元数据"""
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("training_tasks.id", ondelete="SET NULL"), nullable=True)
    dir_name = Column(String(256), unique=True, nullable=False)
    stage = Column(String(8), nullable=False, comment="kg / kt")
    gnn_type = Column(String(32), nullable=False)
    dataset = Column(String(32), nullable=False)
    num_domain = Column(Integer, nullable=False)
    target_domain = Column(Integer, nullable=True)
    random_seed = Column(Integer, nullable=True)
    best_hr = Column(Float, nullable=True)
    best_ndcg = Column(Float, nullable=True)
    best_epoch = Column(Integer, nullable=True)
    total_rounds = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
