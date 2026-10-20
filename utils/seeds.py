"""Deterministic seeding across random / numpy / torch (PDF Part 2.9)."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # noqa: WPS433

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass
