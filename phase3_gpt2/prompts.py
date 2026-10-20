"""Prompt builder for few-shot ICL on codebook-token sequences."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class BuiltPrompt:
    input_ids: list[int]
    labels: list[int]              # -100 except at the answer-token position
    answer_position: int           # index of the answer token in input_ids
    n_support: int
    n_distractors: int
    gold_label: str


def build_prompt(
    bundle,
    support: Sequence[tuple[str, Sequence[int]]],   # (label, indices)
    query_indices: Sequence[int],
    gold_label: str,
    max_length: int = 1024,
    label_to_token: dict[str, int] | None = None,
) -> BuiltPrompt:
    """Assemble token IDs and the answer-only loss mask."""
    sp = bundle.special_token_ids
    sig_ids = bundle.sig_token_ids
    ans_ids = bundle.answer_token_ids

    def tok_of(lbl: str) -> int:
        if label_to_token is not None:
            if lbl not in label_to_token:
                raise KeyError(f"label {lbl!r} not in per-episode label_to_token map")
            return int(label_to_token[lbl])
        if lbl not in ans_ids:
            raise KeyError(f"Unknown label: {lbl}")
        return int(ans_ids[lbl])

    seq: list[int] = [sp["<BOS_SIG>"]]
    for lbl, idxs in support:
        seq.append(sp["<ACT>"])
        seq.append(tok_of(lbl))
        seq.extend(int(sig_ids[i]) for i in idxs)
        seq.append(sp["<SEP>"])

    seq.append(sp["<QUERY>"])
    seq.extend(int(sig_ids[i]) for i in query_indices)
    seq.append(sp["<ANS>"])

    answer_pos = len(seq)
    gold_tok = tok_of(gold_label)
    seq.append(gold_tok)

    if len(seq) > max_length:
        raise ValueError(
            f"Prompt length {len(seq)} > max {max_length}. "
            f"Reduce K-shot or number of distractor classes."
        )

    labels = [-100] * len(seq)
    labels[answer_pos] = gold_tok

    n_support = len(support)
    n_distractors = len({lbl for lbl, _ in support if lbl != gold_label})
    return BuiltPrompt(
        input_ids=seq,
        labels=labels,
        answer_position=answer_pos,
        n_support=n_support,
        n_distractors=n_distractors,
        gold_label=gold_label,
    )


def candidate_label_pool(
    gold_label: str,
    seen_labels: Sequence[str],
    rng,
    max_distractors: int = 6,
    always_include_gold: bool = True,
    distractor_labels: Sequence[str] | None = None,
) -> list[str]:
    """Select up to max_distractors+1 candidate labels for the prompt support."""
    pool = list(distractor_labels) if distractor_labels is not None else list(seen_labels)
    others = [l for l in pool if l != gold_label]
    rng.shuffle(others)
    n = min(max_distractors, len(others))
    picks = others[:n]
    if always_include_gold and gold_label not in picks:
        picks.insert(0, gold_label)
    rng.shuffle(picks)
    return picks


def estimate_prompt_length(
    n_support: int,
    sig_tokens_per_window: int,
) -> int:
    """Number of tokens for a prompt with n_support support windows + 1 query."""
    return 1 + n_support * (3 + sig_tokens_per_window) + (3 + sig_tokens_per_window)
