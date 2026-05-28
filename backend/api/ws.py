"""WebSocket 端点 — 实时训练数据推送"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.ws_manager import manager

logger = logging.getLogger("ws")
router = APIRouter()


@router.websocket("/api/ws/training/{task_id}")
async def training_ws(ws: WebSocket, task_id: int):
    """
    订阅指定训练任务的实时指标 + 日志 + 状态。

    客户端发送:
        {"type": "subscribe", "task_id": <int>}

    服务端推送:
        {"type": "metric", "task_id": <int>, "data": {...}}
        {"type": "log", "task_id": <int>, "data": {...}}
        {"type": "status", "task_id": <int>, "status": "...", "progress": 0.0}
    """
    await manager.connect(ws)
    manager.subscribe_task(ws, task_id)

    try:
        while True:
            # 接收客户端消息（心跳 / 命令）
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "subscribe" and "task_id" in msg:
                    manager.subscribe_task(ws, msg["task_id"])
            except (json.JSONDecodeError, TypeError):
                pass  # 忽略无效消息

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)
