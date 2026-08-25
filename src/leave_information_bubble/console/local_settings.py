"""Allowlisted, atomic persistence for loopback console connection settings."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from leave_information_bubble.config import Settings

_ALLOWED_ENV_KEYS = frozenset(
    {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "BILIBILI_SESSDATA",
        "NGA_COOKIE",
    }
)
_ENV_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
_MAX_VALUE_LENGTH = 20_000


def update_local_settings(path: Path, updates: Mapping[str, str]) -> None:
    """Atomically update supported `.env` values while preserving unrelated lines."""
    normalized = _validate_updates(updates)
    if not normalized:
        return
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    output: list[str] = []
    written: set[str] = set()
    for line in existing:
        match = _ENV_ASSIGNMENT.match(line)
        key = match.group(1) if match is not None else None
        if key not in normalized:
            output.append(line)
            continue
        if key not in written:
            output.append(f'{key}="{_escaped_value(normalized[key])}"')
            written.add(key)
    for key, value in normalized.items():
        if key not in written:
            output.append(f'{key}="{_escaped_value(value)}"')
    content = "\n".join(output) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def local_settings_status(settings: Settings) -> dict[str, object]:
    """Describe usable local connections without returning any secret value."""
    return {
        "provider": {
            "name": "DeepSeek",
            "model": settings.deepseek_model,
            "base_url": settings.deepseek_base_url,
        },
        "configured": {
            "deepseek": bool(settings.deepseek_api_key),
            "bilibili": bool(settings.bilibili_sessdata),
            "nga": bool(settings.nga_cookie),
        },
    }


def _validate_updates(updates: Mapping[str, str]) -> dict[str, str]:
    unknown = sorted(set(updates) - _ALLOWED_ENV_KEYS)
    if unknown:
        raise ValueError(f"unsupported local setting: {unknown[0]}")
    normalized: dict[str, str] = {}
    for key, value in updates.items():
        if not isinstance(value, str):
            raise ValueError(f"local setting {key} must be text")
        if "\r" in value or "\n" in value:
            raise ValueError(f"local setting {key} must be single-line")
        if "${" in value:
            raise ValueError(f"local setting {key} must not contain dotenv interpolation")
        if len(value) > _MAX_VALUE_LENGTH:
            raise ValueError(f"local setting {key} exceeds {_MAX_VALUE_LENGTH} characters")
        normalized[key] = value
    return normalized


def _escaped_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = ["local_settings_status", "update_local_settings"]
