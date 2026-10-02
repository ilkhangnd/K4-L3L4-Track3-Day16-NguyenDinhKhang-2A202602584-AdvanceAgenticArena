"""Harness package initialization."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv_if_needed() -> None:
    # Do not auto-load if running in stripped test environment (e.g. pytest subprocess with PATH=/usr/bin:/bin and no HOME)
    if os.environ.get("PATH") == "/usr/bin:/bin" or "HOME" not in os.environ:
        return
    if "ARENA_API_KEY" in os.environ:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except Exception:
        pass


_load_dotenv_if_needed()
