"""应用配置 - 集中管理所有环境变量"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # 应用
    APP_NAME: str = "FedGCDR 毕设系统"
    DEBUG: bool = True

    # 数据库
    DATABASE_URL: str = "sqlite:///./fedgcdr.db"
    # 生产环境用 PostgreSQL:
    # DATABASE_URL: str = "postgresql://user:pass@localhost:5432/fedgcdr"

    # JWT
    SECRET_KEY: str = "change-this-to-a-secure-random-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480  # 8小时

    # 项目路径
    PROJECT_ROOT: str = "."  # FedGCDR 项目根目录
    CHECKPOINT_DIR: str = "checkpoints"
    OUTPUT_DIR: str = "output"
    DATA_DIR: str = "data"
    EMBEDDING_DIR: str = "embedding"

    # 训练任务
    MAX_CONCURRENT_TRAINING: int = 1  # GPU 并发数
    TRAINING_TIMEOUT_SECONDS: int = 86400  # 24小时超时

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
