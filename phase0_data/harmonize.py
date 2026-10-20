"""Raw-label → unified-label mapping + per-channel z-score normalisation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml


def load_label_map(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_raw_to_unified(label_map: dict) -> dict[tuple[str, str], Optional[str]]:
    """Returns {(dataset, raw_label_lower): unified_label or None}."""
    table: dict[tuple[str, str], Optional[str]] = {}
    for unified, entry in label_map.items():
        if unified.startswith("_"):
            # Null aliases — entry is dict mapping dataset → [raw labels]
            for ds, raws in entry.items():
                for raw in raws:
                    table[(ds, str(raw).lower().strip())] = None
            continue
        aliases = entry.get("aliases", {}) or {}
        for ds, raws in aliases.items():
            for raw in raws:
                table[(ds, str(raw).lower().strip())] = unified
    return table


def unified_label_set(label_map: dict) -> list[str]:
    return [k for k in label_map.keys() if not k.startswith("_")]


def harmonise_labels(
    raw_labels: np.ndarray,
    dataset: str,
    label_map: dict,
) -> np.ndarray:
    """Map raw label strings to unified labels; unknowns -> None."""
    table = build_raw_to_unified(label_map)
    out = np.empty(len(raw_labels), dtype=object)
    misses: set[str] = set()
    for i, raw in enumerate(raw_labels):
        key = (dataset, str(raw).lower().strip())
        if key in table:
            out[i] = table[key]
        else:
            out[i] = None
            misses.add(str(raw))
    if misses:
        print(
            f"[harmonize] {dataset}: {len(misses)} unmapped raw labels "
            f"(samples: {sorted(misses)[:5]}{'...' if len(misses) > 5 else ''})"
        )
    return out


def fit_zscore(X: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std on X[mask] (or all of X)."""
    if mask is not None:
        X_fit = X[mask]
    else:
        X_fit = X
    if X_fit.size == 0:
        return np.zeros(X.shape[-1], dtype=np.float32), np.ones(X.shape[-1], dtype=np.float32)
    mean = X_fit.reshape(-1, X.shape[-1]).mean(axis=0).astype(np.float32)
    std = X_fit.reshape(-1, X.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)
