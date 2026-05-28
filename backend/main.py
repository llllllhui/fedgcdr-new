"""FedGCDR 毕设系统 - FastAPI 主入口"""

import sys
import os

# 确保项目根目录在 Python 路径上
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.core.config import settings
from backend.db.database import engine, Base
from backend.db import models  # noqa: 确保模型被注册
from backend.api import auth, training, checkpoint, recommendation, ws


def init_db():
    """创建所有表（SQLite 适用）"""
    Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    init_db()
    print(f"[OK] Database ready ({settings.DATABASE_URL})")
    print(f"[OK] FedGCDR System - {settings.APP_NAME}")
    yield
    # 关闭时
    print("[INFO] System shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description="FedGCDR 联邦跨域推荐系统 - 毕设管理平台",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - 允许前端开发服务器
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 默认
        "http://localhost:3000",  # 备选
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router)
app.include_router(training.router)
app.include_router(checkpoint.router)
app.include_router(recommendation.router)
app.include_router(ws.router)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME, "version": "1.0.0"}
