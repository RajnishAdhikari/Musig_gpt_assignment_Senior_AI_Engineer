from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATABASE_URL = "postgresql+asyncpg://musicgpt:musicgpt@localhost:5433/musicgpt_search"
DEFAULT_DATASET_URL = (
    "https://lalals.s3.us-east-1.amazonaws.com/"
    "ai_backend_assets/technical_assessment_datasets/song_metadata.json"
)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class Settings:
    database_url: str
    sql_echo: bool
    db_pool_size: int
    db_max_overflow: int
    db_pool_pre_ping: bool
    dataset_url: str
    dataset_path: Path
    dataset_download_timeout_seconds: int
    feedback_flush_interval_seconds: float
    feedback_flush_batch_size: int
    default_search_limit: int
    default_retrieval_limit: int
    embedding_model_name: str
    embedding_device: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            sql_echo=_bool_env("SQL_ECHO", False),
            db_pool_size=_int_env("DB_POOL_SIZE", 5),
            db_max_overflow=_int_env("DB_MAX_OVERFLOW", 10),
            db_pool_pre_ping=_bool_env("DB_POOL_PRE_PING", True),
            dataset_url=os.getenv("DATASET_URL", DEFAULT_DATASET_URL),
            dataset_path=Path(os.getenv("DATASET_PATH", "data/song_metadata.json")),
            dataset_download_timeout_seconds=_int_env("DATASET_DOWNLOAD_TIMEOUT_SECONDS", 120),
            feedback_flush_interval_seconds=_float_env("FEEDBACK_FLUSH_INTERVAL_SECONDS", 5.0),
            feedback_flush_batch_size=_int_env("FEEDBACK_FLUSH_BATCH_SIZE", 25_000),
            default_search_limit=_int_env("DEFAULT_SEARCH_LIMIT", 10),
            default_retrieval_limit=_int_env("DEFAULT_RETRIEVAL_LIMIT", 100),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        )


settings = Settings.from_env()
