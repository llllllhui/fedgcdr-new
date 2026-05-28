"""Checkpoint 管理 API — 扫描文件系统 + CRUD"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User
from backend.auth.deps import get_current_user
from backend.schemas.schemas import CheckpointResponse
from backend.core.config import settings

router = APIRouter(prefix="/api/checkpoints", tags=["Checkpoint管理"])


def _parse_checkpoint_dir(dir_path: Path) -> Optional[dict]:
    """扫描单个 checkpoint 目录，返回结构化元数据"""
    metadata_file = dir_path / "metadata.json"
    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    args = metadata.get("args", {})
    metrics = metadata.get("metrics", {})

    # 从目录名推断阶段
    dir_name = dir_path.name
    if dir_name.startswith("kg_"):
        stage = "kg"
    elif dir_name.startswith("kt_"):
        stage = "kt"
    else:
        stage = metadata.get("stage", "unknown")

    # 统计文件
    file_count = sum(1 for _ in dir_path.iterdir() if _.is_file())
    size_bytes = sum(_.stat().st_size for _ in dir_path.iterdir() if _.is_file())

    # 解析时间戳
    created_at = None
    try:
        ts_str = metadata.get("timestamp", "")
        if ts_str:
            created_at = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        created_at = datetime.fromtimestamp(dir_path.stat().st_mtime)

    return {
        "dir_name": dir_name,
        "stage": stage,
        "gnn_type": args.get("gnn_type", "unknown"),
        "dataset": args.get("dataset", "amazon"),
        "num_domain": int(args.get("num_domain", 0)),
        "target_domain": args.get("target_domain", None),
        "random_seed": int(args.get("random_seed", 0)) if args.get("random_seed") else None,
        "best_hr": metrics.get("max_hr"),
        "best_ndcg": metrics.get("max_ndcg"),
        "best_epoch": metrics.get("best_epoch"),
        "created_at": created_at,
        "full_path": str(dir_path.resolve()),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


# ─── API Endpoints ───


@router.get("/", response_model=list[CheckpointResponse])
def list_checkpoints(
    stage: Optional[str] = Query(None, description="筛选 kg 或 kt"),
    gnn_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """扫描 checkpoints/ 目录，返回所有 checkpoint 列表"""
    checkpoint_dir = Path(settings.CHECKPOINT_DIR).resolve()
    if not checkpoint_dir.exists():
        return []

    results = []
    for item in sorted(checkpoint_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if item.is_dir():
            info = _parse_checkpoint_dir(item)
            if info:
                if stage and info["stage"] != stage:
                    continue
                if gnn_type and info["gnn_type"] != gnn_type:
                    continue
                results.append(CheckpointResponse(**info))

    return results


@router.get("/{dir_name}", response_model=CheckpointResponse)
def get_checkpoint(dir_name: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """获取单个 checkpoint 详情"""
    checkpoint_dir = Path(settings.CHECKPOINT_DIR) / dir_name
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise HTTPException(status_code=404, detail="Checkpoint 目录不存在")

    info = _parse_checkpoint_dir(checkpoint_dir)
    if not info:
        raise HTTPException(status_code=404, detail="无法解析 checkpoint 元数据")

    return CheckpointResponse(**info)


@router.delete("/{dir_name}")
def delete_checkpoint(
    dir_name: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """删除 checkpoint 目录"""
    checkpoint_dir = Path(settings.CHECKPOINT_DIR) / dir_name
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        raise HTTPException(status_code=404, detail="Checkpoint 目录不存在")

    try:
        shutil.rmtree(str(checkpoint_dir))
        return {"message": f"Checkpoint {dir_name} 已删除"}
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
