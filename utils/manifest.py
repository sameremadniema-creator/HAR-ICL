"""Manifest writer — every run produces a JSON tracing config → ckpt → metrics."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def hash_obj(obj: Any) -> str:
    """SHA-256 of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_manifest(
    out_dir: str | Path,
    run_id: str,
    config: dict,
    metrics: dict,
    ckpt_path: str | Path | None = None,
    extra: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "ts": time.time(),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "config_sha256": hash_obj(config),
        "config": config,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "ckpt_sha256": file_sha256(ckpt_path) if ckpt_path and Path(ckpt_path).exists() else None,
        "metrics": metrics,
        "extra": extra or {},
    }
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
