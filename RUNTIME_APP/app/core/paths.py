from __future__ import annotations
import sys
from pathlib import Path

def app_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]

def models_dir() -> Path:
    return app_root() / "models" / "current"

def logs_dir() -> Path:
    return app_root() / "logs"
