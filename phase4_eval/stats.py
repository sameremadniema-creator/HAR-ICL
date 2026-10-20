"""Statistical tests: paired Wilcoxon, bootstrap CI on macro-F1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from har_icl.utils.metrics import bootstrap_ci, macro_f1, paired_wilcoxon


def f1_bootstrap_ci(y_true, y_pred, n: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return bootstrap_ci(
        lambda yt, yp: macro_f1(yt, yp, labels=labels),
        y_true, y_pred, n=n, alpha=alpha, seed=seed,
    )


def aggregate_seeds(paths: Sequence[str]) -> dict:
    """Aggregate test_macro_f1 / test_unseen_f1 / test_seen_f1 across seed runs."""
    arrs: dict[str, list[float]] = {}
    for p in paths:
        s = json.loads(Path(p).read_text())
        for k in ("test_macro_f1", "test_seen_f1", "test_unseen_f1"):
            arrs.setdefault(k, []).append(float(s.get(k, float("nan"))))
    summary = {}
    for k, vs in arrs.items():
        a = np.asarray(vs, dtype=np.float64)
        a = a[~np.isnan(a)]
        summary[k] = {
            "mean": float(a.mean()) if a.size else float("nan"),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n": int(a.size),
            "values": vs,
        }
    return summary


def compare_methods(method_a_summaries: Sequence[str], method_b_summaries: Sequence[str]) -> dict:
    """Paired Wilcoxon between two methods' per-seed test_macro_f1 + test_unseen_f1."""
    a = aggregate_seeds(method_a_summaries)
    b = aggregate_seeds(method_b_summaries)
    out: dict = {"a": a, "b": b, "wilcoxon": {}}
    for k in ("test_macro_f1", "test_seen_f1", "test_unseen_f1"):
        av = np.array(a[k]["values"], dtype=np.float64)
        bv = np.array(b[k]["values"], dtype=np.float64)
        if len(av) == len(bv) and len(av) >= 2:
            out["wilcoxon"][k] = paired_wilcoxon(av, bv)
        else:
            out["wilcoxon"][k] = {"n": min(len(av), len(bv)), "pvalue": None}
    return out
