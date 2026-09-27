"""Append-only experiment records for reproducible comparisons."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def record_experiment(path: str | Path, name: str, config_hash: str, metrics: dict[str, Any], artifacts: dict[str, str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "config_hash": config_hash,
        "metrics": metrics,
        "artifacts": artifacts or {},
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
