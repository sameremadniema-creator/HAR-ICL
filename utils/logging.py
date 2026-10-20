"""Lightweight logging: JSONL stdout + optional W&B offline."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


class JSONLLogger:
    """Append-only newline-delimited JSON logs to stdout AND a file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)
        else:
            self._fh = None

    def log(self, **kv: Any) -> None:
        record = {"ts": time.time(), **kv}
        line = json.dumps(record, default=_to_jsonable)
        print(line, file=sys.stdout, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _to_jsonable(o):
    try:
        import numpy as np  # noqa: WPS433

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def init_wandb(project: str, name: str, config: dict | None = None, offline: bool = True):
    """Initialise W&B in offline mode by default (PDF Part 6.5)."""
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    if not hasattr(wandb, "init"):
        return None
    if offline:
        os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", os.environ.get("WANDB_DIR", "wandb"))
    try:
        return wandb.init(project=project, name=name, config=config or {}, reinit=True)
    except Exception:
        return None
