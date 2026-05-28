"""WebSocket 实时指标推送管理器"""

import asyncio
import json
import logging
from typing import Set, Optional
from fastapi import WebSocket

logger = logging.getLogger("ws")


class ConnectionManager:
    """
    WebSocket 连接管理器 - 广播训练指标到所有订阅者
    """

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._task_subscriptions: dict[int, Set[WebSocket]] = {}  # task_id -> set of Websockets

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self._connections.discard(ws)
        # 从所有订阅中移除
        for subs in self._task_subscriptions.values():
            subs.discard(ws)

    def subscribe_task(self, ws: WebSocket, task_id: int):
        if task_id not in self._task_subscriptions:
            self._task_subscriptions[task_id] = set()
        self._task_subscriptions[task_id].add(ws)

    async def broadcast_metric(self, task_id: int, metric: dict):
        """广播指标数据到订阅了该 task 的所有客户端"""
        message = json.dumps({
            "type": "metric",
            "task_id": task_id,
            "data": metric,
        })
        subs = self._task_subscriptions.get(task_id, set())
        for ws in subs.copy():
            try:
                await ws.send_text(message)
            except Exception:
                subs.discard(ws)

    async def broadcast_log(self, task_id: int, log_entry: dict):
        """广播日志行到订阅了该 task 的所有客户端"""
        message = json.dumps({
            "type": "log",
            "task_id": task_id,
            "data": log_entry,
        })
        subs = self._task_subscriptions.get(task_id, set())
        for ws in subs.copy():
            try:
                await ws.send_text(message)
            except Exception:
                subs.discard(ws)

    async def broadcast_status(self, task_id: int, status: str, progress: float):
        """广播任务状态变更"""
        message = json.dumps({
            "type": "status",
            "task_id": task_id,
            "status": status,
            "progress": progress,
        })
        subs = self._task_subscriptions.get(task_id, set())
        for ws in subs.copy():
            try:
                await ws.send_text(message)
            except Exception:
                subs.discard(ws)


# 全局单例
manager = ConnectionManager()
