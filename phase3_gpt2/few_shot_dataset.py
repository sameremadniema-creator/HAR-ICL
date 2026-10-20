"""K-shot ICL dataset from cached codebook indices."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .prompts import build_prompt, candidate_label_pool


def _load_subset(tokens_dir: Path, split: str, subset: str) -> dict | None:
    p = tokens_dir / f"split_{split}" / f"{subset}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "indices": d["indices"].astype(np.int64),
        "labels": d["labels"].astype(object),
        "subjects": d["subjects"].astype(object),
        "datasets": d["datasets"].astype(object),
    }


class FewShotICLDataset(Dataset):
    """One example per (query, gold_label) pair."""

    def __init__(
        self,
        tokens_dir: str | Path,
        bundle,
        split: str,
        k_shot: int,
        mode: str = "train",
        max_distractors: int = 6,
        max_length: int = 1024,
        seed: int = 0,
        max_examples: int | None = None,
        label_randomization: bool = True,
        distractor_mode: str = "seen",
    ) -> None:
        super().__init__()
        self.distractor_mode = str(distractor_mode)
        self.tokens_dir = Path(tokens_dir)
        meta = json.loads((self.tokens_dir / "meta.json").read_text())
        self.label_set: list[str] = meta["label_set"]
        self.split = split
        self.k_shot = int(k_shot)
        if self.k_shot < 1:
            raise ValueError(
                "k_shot must be >= 1: with k_shot=0 there is no support set to copy from, so each "
                "episode collapses to a single candidate (the gold label) and the metric degenerates "
                "to 1.0. Zero-shot is not meaningful for this in-context-copying protocol."
            )
        self.mode = mode
        self.max_distractors = int(max_distractors)
        self.max_length = int(max_length)
        self.bundle = bundle
        # Meta-ICL: randomise the class->answer-token assignment per episode so the
        self.label_randomization = bool(label_randomization)
        self._answer_token_pool = list(bundle.answer_token_ids.values())

        # Identify unseen labels from split definition baked into.
        train = _load_subset(self.tokens_dir, split, "train")
        val = _load_subset(self.tokens_dir, split, "val")
        test = _load_subset(self.tokens_dir, split, "test")
        support_pool = _load_subset(self.tokens_dir, split, "support_pool")
        if train is None:
            raise FileNotFoundError(f"missing tokens for split {split} under {self.tokens_dir}")
        self.train = train
        self.val = val if val is not None else {"indices": np.empty((0, train["indices"].shape[1]), dtype=np.int64),
                                                "labels": np.empty((0,), dtype=object),
                                                "subjects": np.empty((0,), dtype=object),
                                                "datasets": np.empty((0,), dtype=object)}
        self.test = test if test is not None else dict(self.val)
        self.support_pool = support_pool if support_pool is not None else dict(self.val)

        train_labels = set(self.train["labels"].tolist())
        pool_labels = set(self.support_pool["labels"].tolist())
        self.unseen_labels = sorted(pool_labels - train_labels)
        self.seen_labels = sorted(train_labels)

        # Build per-label index lists for fast K-shot sampling.
        self._label_to_train_idx = _bin_by_label(self.train["labels"])
        self._label_to_support_idx = _bin_by_label(self.support_pool["labels"])

        # Define the *query* pool.
        if mode == "train":
            self.query_indices = self.train["indices"]
            self.query_labels = self.train["labels"]
        elif mode == "val":
            self.query_indices = self.val["indices"]
            self.query_labels = self.val["labels"]
        elif mode == "test":
            self.query_indices = self.test["indices"]
            self.query_labels = self.test["labels"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Row ids into the *current mode's* subset (train/val/test), so we can exclude a
        self.query_row_ids = np.arange(self.query_indices.shape[0])

        if max_examples is not None and len(self.query_labels) > max_examples:
            rng_cap = np.random.default_rng(seed)
            sel = rng_cap.choice(len(self.query_labels), size=max_examples, replace=False)
            self.query_indices = self.query_indices[sel]
            self.query_labels = self.query_labels[sel]
            self.query_row_ids = self.query_row_ids[sel]

        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.query_labels)

    def _sample_support_examples(
        self, gold_label: str, exclude_query_id: int | None = None
    ) -> list[tuple[str, list[int]]]:
        """Sample K support examples per candidate label."""
        if self.distractor_mode == "all_unseen":
            distractor_labels = self.unseen_labels
        elif self.distractor_mode == "mixed":
            distractor_labels = list(self.seen_labels) + list(self.unseen_labels)
        else:
            distractor_labels = self.seen_labels
        candidates = candidate_label_pool(
            gold_label=gold_label,
            seen_labels=self.seen_labels,
            rng=self.rng,
            max_distractors=self.max_distractors,
            always_include_gold=True,
            distractor_labels=distractor_labels,
        )
        # If gold is unseen, ensure it's in candidates.
        if gold_label not in candidates:
            candidates.insert(0, gold_label)
            self.rng.shuffle(candidates)

        support: list[tuple[str, list[int]]] = []
        for lbl in candidates:
            if lbl in self._label_to_train_idx and len(self._label_to_train_idx[lbl]) > 0:
                pool = self.train["indices"]
                pool_idx = self._label_to_train_idx[lbl]
                src = "train"
            elif lbl in self._label_to_support_idx and len(self._label_to_support_idx[lbl]) > 0:
                pool = self.support_pool["indices"]
                pool_idx = self._label_to_support_idx[lbl]
                src = "support_pool"
            else:
                continue
            if exclude_query_id is not None and src == "train":
                pool_idx = [i for i in pool_idx if i != exclude_query_id]
                if not pool_idx:
                    continue
            for _ in range(self.k_shot):
                pick = self.rng.choice(pool_idx)
                support.append((lbl, pool[pick].tolist()))
        self.rng.shuffle(support)
        return support

    def _make_label_map(self, classes: list[str]) -> dict[str, int]:
        """Per-episode class -> answer-token-id map."""
        if not self.label_randomization:
            return {c: int(self.bundle.answer_token_ids[c]) for c in classes}
        toks = self.rng.sample(self._answer_token_pool, len(classes))
        return {c: int(t) for c, t in zip(classes, toks)}

    def __getitem__(self, item: int):
        gold = str(self.query_labels[item])
        query_idx = self.query_indices[item].tolist()
        # Exclude the query window from its own support (only possible in train mode
        eqid = int(self.query_row_ids[item]) if self.mode == "train" else None
        support = self._sample_support_examples(gold, exclude_query_id=eqid)
        # Classes present in this episode = gold + every support class.
        classes = sorted({gold} | {lbl for lbl, _ in support})
        label_to_token = self._make_label_map(classes)
        # Hard cap: drop tail support pairs if prompt exceeds context.
        bp = None
        cur_support = support
        while cur_support:
            try:
                bp = build_prompt(
                    self.bundle, support=cur_support, query_indices=query_idx,
                    gold_label=gold, max_length=self.max_length,
                    label_to_token=label_to_token,
                )
                break
            except ValueError:
                cur_support = cur_support[:-1]
        if bp is None:
            # K=0 zero-shot fallback (gold still needs a token from the map).
            bp = build_prompt(
                self.bundle, support=[], query_indices=query_idx,
                gold_label=gold, max_length=self.max_length,
                label_to_token=label_to_token,
            )
        # Decode info: which tokens are valid answers for THIS episode, and how to map
        present = {gold} | {lbl for lbl, _ in cur_support}
        tok2lbl = {int(label_to_token[c]): c for c in present}
        return {
            "input_ids": torch.tensor(bp.input_ids, dtype=torch.long),
            "labels": torch.tensor(bp.labels, dtype=torch.long),
            "answer_position": bp.answer_position,
            "gold_label": gold,
            "cand_token_ids": sorted(tok2lbl.keys()),
            "tok2lbl": tok2lbl,
        }


def _bin_by_label(labels: np.ndarray) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, lbl in enumerate(labels):
        out.setdefault(str(lbl), []).append(int(i))
    return out


def collate_with_padding(batch: list[dict], pad_token_id: int) -> dict:
    """Right-pad input_ids and labels to the longest in the batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    answer_positions = []
    golds = []
    cand_token_ids = []
    tok2lbl = []
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = 1
        labels[i, :L] = b["labels"]
        answer_positions.append(int(b["answer_position"]))
        golds.append(b["gold_label"])
        cand_token_ids.append(b.get("cand_token_ids", []))
        tok2lbl.append(b.get("tok2lbl", {}))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "answer_positions": answer_positions,
        "gold_labels": golds,
        "cand_token_ids": cand_token_ids,   # per-item list of valid answer token ids
        "tok2lbl": tok2lbl,                 # per-item {token_id: class} for decoding
    }
