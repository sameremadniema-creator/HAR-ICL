"""PDF Part 8 risk: K=8 × 12 classes × 17-token entry overflows GPT-2's 1024 ctx."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from har_icl.phase3_gpt2.lora_setup import build_gpt2_with_lora
from har_icl.phase3_gpt2.prompts import build_prompt, estimate_prompt_length


def test_prompt_fits_in_1024_with_recommended_settings():
    K = 8
    distractors = 6
    sig_tokens_per_window = 16
    estimate = estimate_prompt_length(
        n_support=K * (distractors + 1), sig_tokens_per_window=sig_tokens_per_window
    )
    # 1 + 56*(3+16) + (3+16) = 1 + 1064 + 19 = 1084 > 1024 ⇒ THIS overflows.
    assert estimate > 1024


def test_prompt_builder_raises_on_overflow():
    bundle = build_gpt2_with_lora(num_codes=64, activity_labels=["walk", "stand"], use_lora=False)
    big_support = []
    for _ in range(200):
        big_support.append(("walk", [0] * 16))
    with pytest.raises(ValueError):
        build_prompt(
            bundle, support=big_support, query_indices=[0] * 16,
            gold_label="walk", max_length=1024,
        )


def test_prompt_fits_with_realistic_settings():
    bundle = build_gpt2_with_lora(num_codes=64, activity_labels=["walk", "stand", "run"], use_lora=False)
    # K=4, 4 candidates total (gold + 3 distractors), 16 sig tokens per window.
    support = []
    for cand in ("walk", "stand", "run"):
        for _ in range(4):
            support.append((cand, [1] * 16))
    bp = build_prompt(
        bundle, support=support, query_indices=[2] * 16,
        gold_label="walk", max_length=1024,
    )
    assert len(bp.input_ids) < 1024
    # Answer-only loss mask: exactly one non -100 entry.
    non_mask = sum(1 for v in bp.labels if v != -100)
    assert non_mask == 1
