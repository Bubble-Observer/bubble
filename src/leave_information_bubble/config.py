"""Application settings loaded from environment variables and .env file.

Uses pydantic-settings for typed, validated configuration.
All external service credentials are read here — nowhere else in the codebase.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Typed application configuration.

    All fields can be set via environment variables (e.g. DEEPSEEK_API_KEY)
    or a .env file in the project root.
    """

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    bilibili_sessdata: str = ""
    nga_cookie: str = ""

    data_dir: str = "data"
    output_dir: str = "data/output"
    notebook_path: str = "data/notebook/field_notes.sqlite3"
    asr_enabled: bool = True
    asr_model: str = "small"
    asr_device: str = "auto"
    asr_compute_type: str = "auto"
    asr_local_files_only: bool = True
    asr_language: str = "zh"
    asr_hotwords: str = ""
    asr_max_duration_seconds: int = 3600
    asr_max_audio_bytes: int = 50_000_000
    asr_lock_timeout_seconds: float = 300.0
    asr_timeout_seconds: float = 840.0
    tool_discovery_timeout_seconds: float = 45.0
    tool_hydration_timeout_seconds: float = 90.0
    tool_full_hydration_timeout_seconds: float = 900.0

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    The cache ensures .env is only read once per process.
    """
    return Settings()
