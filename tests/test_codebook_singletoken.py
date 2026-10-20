"""PDF risk #11: each `sig_X` must round-trip to exactly one token id."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from har_icl.phase3_gpt2.lora_setup import build_gpt2_with_lora


def test_sig_tokens_are_single_tokens():
    bundle = build_gpt2_with_lora(
        num_codes=64,
        activity_labels=["walk", "stand", "run"],
        use_lora=False,
    )
    tok = bundle.tokenizer
    for i in (0, 1, 7, 42, 63):
        ids = tok.encode(f"sig_{i}", add_special_tokens=False)
        assert len(ids) == 1, f"sig_{i} encodes to {ids}"
    # Special tokens too.
    for s in ("<BOS_SIG>", "<ACT>", "<SEP>", "<QUERY>", "<ANS>"):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"{s} encodes to {ids}"
    # Activity tokens.
    for lbl in ("walk", "stand", "run"):
        ids = tok.encode(f"act:{lbl}", add_special_tokens=False)
        assert len(ids) == 1
