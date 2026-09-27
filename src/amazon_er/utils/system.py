"""Environment diagnostics that do not import optional GPU packages eagerly."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import psutil

OPTIONAL_PACKAGES = ("faiss", "sentence_transformers", "torch", "transformers", "peft")


def system_report(paths: dict[str, Path] | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_gib": round(memory.total / 2**30, 2),
        "ram_available_gib": round(memory.available / 2**30, 2),
        "optional_packages": {name: importlib.util.find_spec(name) is not None for name in OPTIONAL_PACKAGES},
        "executables": {name: shutil.which(name) for name in ("git", "nvidia-smi", "uroman")},
        "environment": {
            "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if paths:
        report["paths"] = {}
        for name, path in paths.items():
            parent = path if path.exists() else path.parent
            usage = shutil.disk_usage(parent)
            report["paths"][name] = {
                "path": str(path),
                "exists": path.exists(),
                "disk_free_gib": round(usage.free / 2**30, 2),
            }
    return report


def print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))

