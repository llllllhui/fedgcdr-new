"""训练任务 CRUD API"""

import subprocess
import os
import sys
import signal
import threading
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import TrainingTask, TrainingStatus, TrainingLog, MetricRecord, Checkpoint, User
from backend.auth.deps import get_current_user
from backend.schemas.schemas import (
    TrainingConfig, TrainingTaskResponse, TrainingTaskDetail, LogEntry, MetricPoint,
)
from backend.core.config import settings

router = APIRouter(prefix="/api/training", tags=["训练任务"])

# 全局运行中的进程表 {task_id: subprocess.Popen}
_running_processes: dict[int, subprocess.Popen] = {}
_lock = threading.Lock()


def _build_main_args(config: TrainingConfig) -> list[str]:
    """将 API 配置转换为 main.py 命令行参数"""
    args = [
        sys.executable, "main.py",
        "--gnn_type", config.gnn_type,
        "--dataset", config.dataset,
        "--num_domain", str(config.num_domain),
        "--target_domain", str(config.target_domain),
        "--embedding_size", str(config.embedding_size),
        "--round_gat", str(config.round_gat),
        "--round_ft", str(config.round_ft),
        "--lr_gnn", str(config.lr_gnn),
        "--lr_mf", str(config.lr_mf),
        "--dp", str(config.dp),
        "--eps", str(config.eps),
        "--random_seed", str(config.random_seed),
        "--local_epoch", str(config.local_epoch),
        "--user_batch", str(config.user_batch),
        "--weight_decay", str(config.weight_decay),
        "--num_negative", str(config.num_negative),
        "--use_knowledge_gate", str(config.knowledge_gate),
        "--knowledge_gate_threshold", str(config.knowledge_gate_threshold),
    ]
    return args


def _parse_metric_line(line: str) -> Optional[dict]:
    """从日志行解析指标数据"""
    # 匹配: [DomainName Phase Round N] hr_5 = X.XXXX, ndcg_5 = X.XXXX, hr_10 = X.XXXX, ndcg_10 = X.XXXX
    pattern = r'\[([^\]]+)\]\s+hr_5\s*=\s*([\d.]+),\s*ndcg_5\s*=\s*([\d.]+),\s*hr_10\s*=\s*([\d.]+),\s*ndcg_10\s*=\s*([\d.]+)'
    m = re.search(pattern, line)
    if m:
        header = m.group(1)
        # header: "DomainName Phase Round N"
        parts = header.split()
        if len(parts) >= 4:
            domain = parts[0]
            phase = parts[1]
            round_idx = int(parts[3])
            return {
                "domain": domain,
                "phase": phase,
                "round": round_idx,
                "hr_5": float(m.group(2)),
                "ndcg_5": float(m.group(3)),
                "hr_10": float(m.group(4)),
                "ndcg_10": float(m.group(5)),
            }
    return None


def _parse_final_metrics(line: str) -> Optional[dict]:
    """从输出文件末尾解析最终指标"""
    pattern = r'hr_5\s*=\s*([\d.]+),\s*ndcg_5\s*=\s*([\d.]+),\s*hr_10\s*=\s*([\d.]+),\s*ndcg_10\s*=\s*([\d.]+)'
    m = re.search(pattern, line)
    if m:
        return {
            "hr_5": float(m.group(1)),
            "ndcg_5": float(m.group(2)),
            "hr_10": float(m.group(3)),
            "ndcg_10": float(m.group(4)),
        }
    return None


def _run_training_thread(task_id: int, config: TrainingConfig, db_url: str):
    """在后台线程中执行训练任务"""
    import asyncio
    from backend.db.database import SessionLocal, engine
    from backend.db.models import Base  # noqa: ensure models loaded
    from backend.api.ws_manager import manager as ws_manager

    db = SessionLocal()
    try:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if not task:
            return

        # 更新状态为 running
        task.status = TrainingStatus.RUNNING.value
        task.started_at = datetime.utcnow()
        db.commit()

        args = _build_main_args(config)
        cwd = Path(settings.PROJECT_ROOT).resolve()

        # 启动子进程
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with _lock:
            _running_processes[task_id] = proc

        task.pid = proc.pid
        db.commit()

        # 创建事件循环用于 WebSocket 广播
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = None

        # 广播任务开始
        if loop:
            try:
                loop.run_until_complete(
                    ws_manager.broadcast_status(task_id, "running", 0.0)
                )
            except Exception:
                pass

        step_counter = 0
        # 读取输出行
        for line in proc.stdout:
            line = line.rstrip("\n\r")
            if not line:
                continue

            step_counter += 1

            # 保存日志
            log = TrainingLog(
                task_id=task_id,
                level="INFO",
                message=line,
            )
            db.add(log)

            # WebSocket 广播日志（non-blocking fire-and-forget）
            if loop:
                try:
                    loop.run_until_complete(
                        ws_manager.broadcast_log(task_id, {
                            "id": 0,
                            "level": "INFO",
                            "message": line,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    )
                except Exception:
                    pass

            # 解析指标行
            metric = _parse_metric_line(line)
            if metric:
                record = MetricRecord(
                    task_id=task_id,
                    step=step_counter,
                    stage=metric["phase"],
                    domain=metric["domain"],
                    round=metric["round"],
                    hr_5=metric["hr_5"],
                    ndcg_5=metric["ndcg_5"],
                    hr_10=metric["hr_10"],
                    ndcg_10=metric["ndcg_10"],
                )
                db.add(record)
                db.commit()

                # WebSocket 广播指标
                if loop:
                    try:
                        loop.run_until_complete(
                            ws_manager.broadcast_metric(task_id, metric)
                        )
                    except Exception:
                        pass

        proc.wait()

        # 解析最终指标
        if proc.returncode == 0:
            task.status = TrainingStatus.COMPLETED.value
            # 尝试从最后一条日志解析最终指标
            last_logs = (
                db.query(TrainingLog)
                .filter(TrainingLog.task_id == task_id)
                .order_by(TrainingLog.id.desc())
                .limit(10)
                .all()
            )
            for log in reversed(last_logs):
                final_metric = _parse_final_metrics(log.message)
                if final_metric:
                    task.best_hr5 = final_metric.get("hr_5")
                    task.best_hr10 = final_metric.get("hr_10")
                    task.best_ndcg5 = final_metric.get("ndcg_5")
                    task.best_ndcg10 = final_metric.get("ndcg_10")
                    break
        else:
            task.status = TrainingStatus.FAILED.value
            if loop:
                try:
                    loop.run_until_complete(
                        ws_manager.broadcast_status(task_id, "failed", 100.0)
                    )
                except Exception:
                    pass

        task.progress = 100.0
        task.finished_at = datetime.utcnow()
        if task.started_at:
            task.duration_seconds = (task.finished_at - task.started_at).total_seconds()
        db.commit()

        # 广播完成状态
        if loop:
            try:
                loop.run_until_complete(
                    ws_manager.broadcast_status(task_id, task.status, 100.0)
                )
            except Exception:
                pass

    except Exception as e:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if task:
            task.status = TrainingStatus.FAILED.value
            task.finished_at = datetime.utcnow()
            log = TrainingLog(task_id=task_id, level="ERROR", message=f"系统错误: {str(e)}")
            db.add(log)
            db.commit()
    finally:
        with _lock:
            _running_processes.pop(task_id, None)
        db.close()


# ─── API Endpoints ───


@router.get("/", response_model=list[TrainingTaskResponse])
def list_tasks(
    status: Optional[str] = Query(None, description="筛选状态"),
    gnn_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(TrainingTask)
    if status:
        q = q.filter(TrainingTask.status == status)
    if gnn_type:
        q = q.filter(TrainingTask.gnn_type == gnn_type)
    return q.order_by(TrainingTask.created_at.desc()).limit(limit).all()


@router.post("/", response_model=TrainingTaskResponse)
def create_task(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 检查并发限制
    running_count = (
        db.query(TrainingTask)
        .filter(TrainingTask.status == TrainingStatus.RUNNING.value)
        .count()
    )
    if running_count >= settings.MAX_CONCURRENT_TRAINING:
        raise HTTPException(
            status_code=429,
            detail=f"已达最大并发训练数 ({settings.MAX_CONCURRENT_TRAINING})，请等待当前任务完成",
        )

    task = TrainingTask(
        name=config.name or f"{config.gnn_type.upper()} {config.num_domain}domains",
        description=config.description,
        creator_id=current_user.id,
        gnn_type=config.gnn_type,
        dataset=config.dataset,
        num_domain=config.num_domain,
        target_domain=config.target_domain,
        embedding_size=config.embedding_size,
        round_gat=config.round_gat,
        round_ft=config.round_ft,
        lr_gnn=config.lr_gnn,
        lr_mf=config.lr_mf,
        dp=config.dp,
        eps=config.eps,
        random_seed=config.random_seed,
        local_epoch=config.local_epoch,
        user_batch=config.user_batch,
        weight_decay=config.weight_decay,
        num_negative=config.num_negative,
        knowledge_gate=config.knowledge_gate,
        knowledge_gate_threshold=config.knowledge_gate_threshold,
        extra_params=config.extra_params,
        status=TrainingStatus.PENDING.value,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # 启动后台训练线程
    background_tasks.add_task(
        _run_training_thread,
        task.id,
        config,
        settings.DATABASE_URL,
    )

    # 立即更新状态
    db.refresh(task)
    return task


@router.get("/{task_id}", response_model=TrainingTaskDetail)
def get_task(task_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@router.post("/{task_id}/cancel")
def cancel_task(task_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status != TrainingStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="只能取消运行中的任务")

    with _lock:
        proc = _running_processes.get(task_id)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            _running_processes.pop(task_id, None)

    task.status = TrainingStatus.CANCELLED.value
    task.finished_at = datetime.utcnow()
    db.commit()
    return {"message": "任务已取消"}


@router.get("/{task_id}/metrics", response_model=list[MetricPoint])
def get_task_metrics(task_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return (
        db.query(MetricRecord)
        .filter(MetricRecord.task_id == task_id)
        .order_by(MetricRecord.step)
        .all()
    )


@router.get("/{task_id}/logs", response_model=list[LogEntry])
def get_task_logs(
    task_id: int,
    after_id: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    q = db.query(TrainingLog).filter(
        TrainingLog.task_id == task_id,
        TrainingLog.id > after_id,
    )
    return q.order_by(TrainingLog.id).limit(limit).all()


@router.delete("/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status == TrainingStatus.RUNNING.value:
        raise HTTPException(status_code=400, detail="请先取消运行中的任务")

    db.delete(task)
    db.commit()
    return {"message": "任务已删除"}
