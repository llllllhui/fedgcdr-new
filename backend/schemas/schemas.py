"""Pydantic 请求/响应 Schema"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from datetime import datetime


# ─── Auth ───

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    role: str = "viewer"


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Training Task ───

class TrainingConfig(BaseModel):
    """创建训练任务的配置"""
    name: Optional[str] = None
    description: Optional[str] = None
    gnn_type: str = Field(..., pattern=r"^(gat|lightgcn|graphsage|gcn)$")
    dataset: str = "amazon"
    num_domain: int = Field(default=4, ge=2, le=16)
    target_domain: int = Field(default=1, ge=-1)
    embedding_size: int = Field(default=16, ge=8, le=128)
    round_gat: int = Field(default=30, ge=1, le=500)
    round_ft: int = Field(default=60, ge=1, le=500)
    lr_gnn: float = Field(default=0.001, ge=1e-6, le=1.0)
    lr_mf: float = Field(default=0.005, ge=1e-6, le=1.0)
    dp: bool = True
    eps: float = Field(default=8.0, ge=0.1, le=100.0)
    random_seed: int = Field(default=42, ge=0, le=999999)
    local_epoch: int = Field(default=3, ge=1, le=50)
    user_batch: int = Field(default=16, ge=1, le=512)
    weight_decay: float = Field(default=1e-4, ge=0, le=1.0)
    num_negative: int = Field(default=4, ge=1, le=50)
    knowledge_gate: bool = True
    knowledge_gate_threshold: float = Field(default=0.5, ge=0, le=1.0)
    extra_params: Optional[dict] = None


class TrainingTaskResponse(BaseModel):
    id: int
    name: Optional[str] = None
    description: Optional[str] = None
    gnn_type: str
    dataset: str
    num_domain: int
    target_domain: int
    status: str
    progress: float
    creator_id: Optional[int] = None
    output_file: Optional[str] = None
    checkpoint_paths: Optional[list] = None
    best_hr5: Optional[float] = None
    best_hr10: Optional[float] = None
    best_ndcg5: Optional[float] = None
    best_ndcg10: Optional[float] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingTaskDetail(TrainingTaskResponse):
    """包含配置参数详细信息的响应"""
    embedding_size: int
    round_gat: int
    round_ft: int
    lr_gnn: float
    lr_mf: float
    dp: bool
    eps: float
    random_seed: int
    local_epoch: int
    user_batch: int
    weight_decay: float
    extra_params: Optional[dict] = None
    logs: List["LogEntry"] = []


class LogEntry(BaseModel):
    id: int
    level: str
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True


# ─── Metric ───

class MetricPoint(BaseModel):
    step: int
    stage: str
    domain: str
    round: int
    hr_5: Optional[float] = None
    ndcg_5: Optional[float] = None
    hr_10: Optional[float] = None
    ndcg_10: Optional[float] = None

    class Config:
        from_attributes = True


# ─── Checkpoint ───

class CheckpointResponse(BaseModel):
    id: Optional[int] = None
    dir_name: str
    stage: str
    gnn_type: str
    dataset: str
    num_domain: int
    target_domain: Optional[int] = None
    random_seed: Optional[int] = None
    best_hr: Optional[float] = None
    best_ndcg: Optional[float] = None
    best_epoch: Optional[int] = None
    created_at: Optional[datetime] = None
    # 文件系统信息
    full_path: Optional[str] = None
    file_count: int = 0
    size_bytes: int = 0

    class Config:
        from_attributes = True
