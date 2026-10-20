"""SLURM-friendly checkpointing: SIGUSR1 handler for --signal=B:USR1@300 graceful save."""
from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Callable


class GracefulCheckpoint:
    """Install a SIGUSR1 handler that flips a flag the train loop checks."""

    def __init__(self) -> None:
        self.should_save_and_exit = False
        try:
            signal.signal(signal.SIGUSR1, self._on_signal)
        except (AttributeError, ValueError):
            # SIGUSR1 unavailable on Windows / non-main thread — fine for local dev.
            pass

    def _on_signal(self, signum, frame) -> None:
        self.should_save_and_exit = True

    def check(self, save_fn: Callable[[], None]) -> None:
        if self.should_save_and_exit:
            save_fn()
            print("[checkpointing] SIGUSR1 received — saved and exiting.", flush=True)
            sys.exit(0)


def save_torch_checkpoint(state: dict, path: str | Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_torch_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    import torch

    return torch.load(str(path), map_location=map_location, weights_only=False)
