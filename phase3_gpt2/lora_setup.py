"""Augment GPT-2 with sig_X tokens + activity-name tokens + LoRA on c_attn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SPECIAL_TOKENS = ("<BOS_SIG>", "<ACT>", "<SEP>", "<QUERY>", "<ANS>", "<PAD>")


@dataclass
class TokenisedModelBundle:
    model: object   # GPT-2 (+ LoRA) HF model
    tokenizer: object
    sig_token_ids: list[int]
    special_token_ids: dict[str, int]
    label_token_ids: dict[str, list[int]]   # label string -> token id sequence
    answer_token_ids: dict[str, int]


def build_gpt2_with_lora(
    num_codes: int,
    activity_labels: Sequence[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    base_model: str = "gpt2",
    target_modules: Sequence[str] = ("c_attn",),
    use_lora: bool = True,
) -> TokenisedModelBundle:
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError as e:
        raise RuntimeError("transformers not installed. `pip install transformers`.") from e

    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    model = GPT2LMHeadModel.from_pretrained(base_model)

    # 1.
    sig_strs = [f"sig_{i}" for i in range(num_codes)]
    tokenizer.add_tokens(list(SPECIAL_TOKENS), special_tokens=True)
    tokenizer.add_tokens(sig_strs, special_tokens=False)

    # 2.
    act_strs = [f"act:{lbl}" for lbl in activity_labels]
    tokenizer.add_tokens(act_strs, special_tokens=False)

    # Set a pad token if missing (GPT-2's tokenizer has none by default).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<PAD>"

    model.resize_token_embeddings(len(tokenizer))

    # Verify that each newly added token round-trips to exactly one ID.
    def _id_of(tok: str) -> int:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) != 1:
            raise AssertionError(f"Token {tok!r} encodes to {len(ids)} ids, expected 1.")
        return ids[0]

    sig_ids = [_id_of(s) for s in sig_strs]
    special_ids = {s: _id_of(s) for s in SPECIAL_TOKENS}
    answer_ids = {lbl: _id_of(f"act:{lbl}") for lbl in activity_labels}
    # Natural-language label tokens (for free-form prompt readability).
    label_token_ids = {lbl: tokenizer.encode(lbl, add_special_tokens=False) for lbl in activity_labels}

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise RuntimeError("peft not installed. `pip install peft`.") from e
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            task_type="CAUSAL_LM",
            # CRITICAL: the newly added sig_* and act:* tokens get random embeddings on
            modules_to_save=["wte", "lm_head"],
        )
        model = get_peft_model(model, cfg)

    return TokenisedModelBundle(
        model=model,
        tokenizer=tokenizer,
        sig_token_ids=sig_ids,
        special_token_ids=special_ids,
        label_token_ids=label_token_ids,
        answer_token_ids=answer_ids,
    )


def trainable_parameter_count(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total
