"""Metrics smoke tests — verify shapes, ranges, edge cases."""
from __future__ import annotations

import numpy as np

from har_icl.utils.metrics import (
    bootstrap_ci,
    codebook_perplexity,
    codebook_usage,
    macro_f1,
    paired_wilcoxon,
    per_class_f1,
    token_entropy,
)


def test_macro_f1_perfect():
    y = ["a", "b", "a", "c"]
    assert macro_f1(y, y) == 1.0


def test_per_class_f1_keys():
    y = ["a", "b", "a"]
    p = ["a", "a", "a"]
    out = per_class_f1(y, p, labels=["a", "b"])
    assert set(out.keys()) == {"a", "b"}


def test_codebook_perplexity_uniform_equals_num_codes():
    idx = np.arange(1024)
    perp = codebook_perplexity(idx, num_codes=1024)
    assert perp > 1023.9


def test_codebook_perplexity_degenerate_equals_one():
    idx = np.zeros(1024, dtype=np.int64)
    assert abs(codebook_perplexity(idx, 1024) - 1.0) < 1e-6


def test_token_entropy_bounds():
    idx = np.arange(64)
    bits = token_entropy(idx, num_codes=64)
    assert 5.99 < bits <= 6.0


def test_codebook_usage_struct():
    idx = np.array([0, 1, 1, 2, 2, 2])
    u = codebook_usage(idx, num_codes=8)
    assert u["used"] == 3
    assert 0.0 < u["usage_frac"] <= 1.0
    assert 1.0 <= u["perplexity"] <= 8.0


def test_bootstrap_ci_keys_and_range():
    y = np.array(["a", "b", "a", "b", "a"])
    yp = np.array(["a", "b", "b", "b", "a"])
    out = bootstrap_ci(lambda yt, yh: macro_f1(yt, yh), y, yp, n=100, seed=1)
    assert {"point", "lo", "hi", "n_resamples"}.issubset(out.keys())
    assert 0.0 <= out["lo"] <= out["point"] <= out["hi"] <= 1.0


def test_paired_wilcoxon_returns_pvalue():
    a = [0.5, 0.6, 0.7, 0.55, 0.65]
    b = [0.3, 0.4, 0.5, 0.45, 0.5]
    out = paired_wilcoxon(a, b)
    assert "pvalue" in out
    assert 0.0 <= out["pvalue"] <= 1.0
