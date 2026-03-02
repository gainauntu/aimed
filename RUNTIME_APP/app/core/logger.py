from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from app.core.paths import logs_dir

def init_logger(name: str = "pill_ai"):
    logs_dir().mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir() / f"{name}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if getattr(logger, "_inited", False):
        return logger
    logger._inited = True

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    logger.info(f"Log file: {log_path}")
    return logger
