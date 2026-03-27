from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.domain.models import DecisionResult


class AuditLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, result: DecisionResult) -> None:
        payload = result.to_dict()
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
