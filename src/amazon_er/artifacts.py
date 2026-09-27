"""Atomic files, input fingerprints, and content-addressed stage manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


def file_fingerprint(path: str | Path, hash_limit: int = 64 * 1024 * 1024) -> dict[str, Any]:
    item = Path(path).resolve()
    stat = item.stat()
    result: dict[str, Any] = {
        "path": str(item),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size <= hash_limit:
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass
class StageManifest:
    stage: str
    status: str
    started_at: float
    finished_at: float | None
    config_sha256: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    environment: dict[str, Any]
    details: dict[str, Any]


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }


@contextmanager
def stage_manifest(
    path: str | Path,
    stage: str,
    config_sha256: str,
    inputs: Iterable[str | Path] = (),
    details: dict[str, Any] | None = None,
) -> Iterator[StageManifest]:
    manifest = StageManifest(
        stage=stage,
        status="running",
        started_at=time.time(),
        finished_at=None,
        config_sha256=config_sha256,
        inputs=[file_fingerprint(item) for item in inputs],
        outputs=[],
        environment=_environment(),
        details=details or {},
    )
    atomic_json(path, asdict(manifest))
    try:
        yield manifest
    except BaseException as exc:
        manifest.status = "failed"
        manifest.finished_at = time.time()
        manifest.details["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(path, asdict(manifest))
        raise
    else:
        manifest.status = "complete"
        manifest.finished_at = time.time()
        atomic_json(path, asdict(manifest))


def record_outputs(manifest: StageManifest, paths: Iterable[str | Path]) -> None:
    manifest.outputs = [file_fingerprint(path) for path in paths]

