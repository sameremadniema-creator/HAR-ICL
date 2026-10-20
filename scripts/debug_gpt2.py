#!/usr/bin/env python3
"""debug_gpt2.py — login-node smoke test for the GPT-2 ICL phase BEFORE the multi-hour job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from har_icl.phase3_gpt2.lora_setup import build_gpt2_with_lora
from har_icl.phase3_gpt2.few_shot_dataset import FewShotICLDataset, collate_with_padding


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--split", default="C")
    ap.add_argument("--k_shot", type=int, default=4)
    ap.add_argument("--max_distractors", type=int, default=6)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--n", type=int, default=200, help="episodes to scan for length stats")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--no_lora", action="store_true", help="skip LoRA (faster build) for the length scan")
    args = ap.parse_args()

    tokens_dir = Path(args.tokens)
    meta = json.loads((tokens_dir / "meta.json").read_text())
    label_set = list(meta["label_set"])
    num_codes = int(meta["num_codes"])
    print(f"[debug_gpt2] tokens={tokens_dir.name} num_codes={num_codes} labels={len(label_set)}")

    bundle = build_gpt2_with_lora(
        num_codes=num_codes, activity_labels=label_set,
        rank=args.rank, alpha=2 * args.rank, dropout=0.05,
        target_modules=("c_attn",), use_lora=not args.no_lora,
    )

    # test mode = hardest: queries may be unseen, support spans train + support_pool.
    ds = FewShotICLDataset(
        tokens_dir=tokens_dir, bundle=bundle, split=args.split, k_shot=args.k_shot,
        mode="test", max_distractors=args.max_distractors, max_length=args.max_length,
        seed=42, label_randomization=True,
    )
    sig_per_win = int(ds.test["indices"].shape[1])
    per_example = 3 + sig_per_win
    full_candidates = min(args.max_distractors, len(ds.seen_labels) - 1) + 1
    expect_len = 1 + full_candidates * args.k_shot * per_example + per_example
    print(f"[debug_gpt2] sig_tokens/window={sig_per_win}  full episode "
          f"={full_candidates} classes x{args.k_shot} -> expected len={expect_len} "
          f"(context {args.max_length})")
    print(f"[debug_gpt2] seen={len(ds.seen_labels)} unseen={ds.unseen_labels}")

    # 1.
    n = min(args.n, len(ds))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ds), size=n, replace=False)
    lengths, truncated = [], 0
    for i in sel:
        item = ds[int(i)]
        L = int(item["input_ids"].numel())
        lengths.append(L)
        # implied support windows = (L - 1 - per_example) / per_example
        implied = round((L - 1 - per_example) / per_example)
        if implied < full_candidates * args.k_shot:
            truncated += 1
    lengths = np.array(lengths)
    print(f"[debug_gpt2] prompt length over {n} episodes: "
          f"min={lengths.min()} mean={lengths.mean():.0f} max={lengths.max()}")
    ok_len = lengths.max() <= args.max_length
    print(f"[debug_gpt2] fits context (max<= {args.max_length}): {ok_len}  | "
          f"episodes with dropped support: {truncated}/{n} "
          f"({100*truncated/n:.1f}%)")

    # 2.
    fb_ok = True
    if not args.no_lora:
        pad_id = int(bundle.tokenizer.pad_token_id)
        batch = collate_with_padding([ds[int(sel[j])] for j in range(min(2, n))], pad_token_id=pad_id)
        model = bundle.model.train()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        out.loss.backward()
        fb_ok = bool(torch.isfinite(out.loss))
        print(f"[debug_gpt2] forward+backward loss={float(out.loss):.4f} finite={fb_ok} "
              f"seq_len={batch['input_ids'].shape[1]}")
        # 3.
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        ans = batch["answer_positions"][0]
        cand = batch["cand_token_ids"][0]
        row = logits[0, ans - 1]
        mask = torch.full_like(row, float("-inf"))
        for tid in cand:
            mask[tid] = 0.0
        pred = batch["tok2lbl"][0].get(int(torch.argmax(row + mask)), "<unk>")
        print(f"[debug_gpt2] decode ok: gold={batch['gold_labels'][0]} pred={pred} "
              f"(random init, value irrelevant)")

    ok = ok_len and fb_ok and truncated == 0
    print("\nGPT2 DEBUG PASSED — safe to submit." if ok
          else "\nGPT2 DEBUG: review above (truncation or overflow) before submitting.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
