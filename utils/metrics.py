"""Metrics: macro/per-class F1, codebook perplexity, token entropy, bootstrap CI."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def macro_f1(y_true, y_pred, labels: Sequence | None = None) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_f1(y_true, y_pred, labels: Sequence) -> dict:
    from sklearn.metrics import f1_score

    scores = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return {str(lbl): float(s) for lbl, s in zip(labels, scores)}


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def codebook_perplexity(indices, num_codes: int) -> float:
    """Exp of the Shannon entropy (in nats) over the code histogram."""
    indices = np.asarray(indices).reshape(-1)
    if indices.size == 0:
        return 0.0
    counts = np.bincount(indices, minlength=num_codes).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs > 0
    entropy = -float(np.sum(probs[nz] * np.log(probs[nz])))
    return float(np.exp(entropy))


def token_entropy(indices, num_codes: int) -> float:
    """Shannon entropy in bits."""
    indices = np.asarray(indices).reshape(-1)
    if indices.size == 0:
        return 0.0
    counts = np.bincount(indices, minlength=num_codes).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs > 0
    return float(-np.sum(probs[nz] * np.log2(probs[nz])))


def codebook_usage(indices, num_codes: int) -> dict:
    indices = np.asarray(indices).reshape(-1)
    counts = np.bincount(indices, minlength=num_codes)
    used = int((counts > 0).sum())
    return {
        "num_codes": int(num_codes),
        "used": used,
        "usage_frac": used / max(num_codes, 1),
        "perplexity": codebook_perplexity(indices, num_codes),
        "entropy_bits": token_entropy(indices, num_codes),
    }


def bootstrap_ci(
    metric_fn: Callable,
    y_true,
    y_pred,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n_resamples": n}
    point = float(metric_fn(y_true, y_pred))
    samples = np.empty(n, dtype=np.float64)
    idx_pool = np.arange(len(y_true))
    for i in range(n):
        idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        samples[i] = float(metric_fn(y_true[idx], y_pred[idx]))
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return {"point": point, "lo": lo, "hi": hi, "n_resamples": n}


def overfit_report(history, loss_key: str = "train_loss", val_loss_key: str = "val_loss",
                   monitor_key: str = "val_blended_f1") -> dict:
    """Flag over/underfitting from a per-epoch training history and print a one-line verdict."""
    try:
        tl = [float(h[loss_key]) for h in history if loss_key in h]
        vl = [float(h[val_loss_key]) for h in history if val_loss_key in h]
        mon = [float(h[monitor_key]) for h in history if monitor_key in h]
        if len(vl) < 3 or len(tl) < 3:
            return {"verdict": "n/a", "reason": "too few epochs"}
        val_min = min(vl); val_rise = vl[-1] - val_min
        reasons = []
        if val_rise > 0.10:
            reasons.append(f"val_loss rose +{val_rise:.2f} above its min")
        if mon:
            mon_peak = max(mon); mon_drop = mon_peak - mon[-1]
            if mon_drop > 0.05:
                reasons.append(f"{monitor_key} fell {mon_drop:.2f} below peak")
        tv_gap = vl[-1] - tl[-1]
        if tl[-1] < 0.05 and tv_gap > 0.30:
            reasons.append(f"train_loss~0 with train/val gap {tv_gap:.2f} (memorisation)")
        underfit = bool(mon) and mon[-1] >= mon_peak - 1e-6 and len(mon) >= 3 and mon[-1] > mon[-3]
        verdict = "OVERFIT" if reasons else ("STILL-IMPROVING (underfit?)" if underfit else "ok")
        diag = {"verdict": verdict, "reasons": reasons, "epochs": len(vl),
                "train_loss_final": round(tl[-1], 4), "val_loss_min": round(val_min, 4),
                "val_loss_final": round(vl[-1], 4),
                "monitor_peak": round(max(mon), 4) if mon else None,
                "monitor_final": round(mon[-1], 4) if mon else None}
        print(f"[overfit-check] {verdict} | epochs={diag['epochs']} train_final={diag['train_loss_final']} "
              f"val_min={diag['val_loss_min']} val_final={diag['val_loss_final']} "
              f"monitor_peak={diag['monitor_peak']} monitor_final={diag['monitor_final']}"
              + (f" | {'; '.join(reasons)}" if reasons else ""), flush=True)
        return diag
    except Exception as e:  # noqa: BLE001
        print(f"[overfit-check] skipped ({e})", flush=True)
        return {"verdict": "error", "reason": str(e)}


def paired_wilcoxon(a, b) -> dict:
    """Paired Wilcoxon signed-rank test across seeds (PDF Part 2.9)."""
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or len(a) < 2:
        return {"stat": float("nan"), "pvalue": float("nan"), "n": int(len(a))}
    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        return {"stat": float("nan"), "pvalue": 1.0, "n": int(len(a))}
    return {"stat": float(stat), "pvalue": float(p), "n": int(len(a))}
