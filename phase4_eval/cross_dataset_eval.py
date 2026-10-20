"""Cross-dataset evaluation: train on UCI-HAR+PAMAP2+MHealth, test on USC-HAD."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from har_icl.phase3_gpt2.few_shot_dataset import FewShotICLDataset, collate_with_padding
from har_icl.phase3_gpt2.lora_setup import build_gpt2_with_lora
from har_icl.utils.metrics import macro_f1, per_class_f1
from har_icl.utils.seeds import set_global_seed

from .stats import f1_bootstrap_ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tokens", type=str, required=True)
    parser.add_argument("--split", type=str, default="C")
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held_out_dataset", type=str, default="usc_had")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens_dir = Path(args.tokens)
    meta = json.loads((tokens_dir / "meta.json").read_text())
    label_set = list(meta["label_set"])
    num_codes = int(meta["num_codes"])

    train_summary_path = Path(args.ckpt).parent / "summary.json"
    cfg = json.loads(train_summary_path.read_text())["cfg"] if train_summary_path.exists() else {
        "model": {"context_length": 1024},
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["c_attn"]},
        "prompt": {"k_shot": args.k_shot, "max_distractors": 6},
        "train": {"batch_size": 16},
    }

    bundle = build_gpt2_with_lora(
        num_codes=num_codes, activity_labels=label_set,
        rank=int(cfg["lora"]["rank"]), alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        target_modules=cfg["lora"]["target_modules"],
        use_lora=False,
    )
    bundle.model.resize_token_embeddings(len(bundle.tokenizer))
    try:
        from peft import PeftModel  # type: ignore

        bundle.model = PeftModel.from_pretrained(bundle.model, args.ckpt)
    except Exception:
        pass
    bundle.model.to(device).eval()

    # Build the test dataset, then filter to held-out dataset only.
    ds = FewShotICLDataset(
        tokens_dir=tokens_dir, bundle=bundle, split=args.split,
        k_shot=int(args.k_shot), mode="test",
        max_distractors=int(cfg["prompt"]["max_distractors"]),
        max_length=int(cfg["model"]["context_length"]),
        seed=args.seed,
    )
    test_subset = np.load(tokens_dir / f"split_{args.split}" / "test.npz", allow_pickle=True)
    datasets_arr = test_subset["datasets"].astype(object)
    keep = np.array([d == args.held_out_dataset for d in datasets_arr])
    if not keep.any():
        print(f"[cross_dataset] no test windows from dataset={args.held_out_dataset!r} found.")
        return
    ds.query_indices = ds.query_indices[keep]
    ds.query_labels = ds.query_labels[keep]
    print(f"[cross_dataset] {keep.sum()} test windows from {args.held_out_dataset}")

    pad_id = int(bundle.tokenizer.pad_token_id)
    collate = partial(collate_with_padding, pad_token_id=pad_id)
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False, num_workers=0, collate_fn=collate)
    answer_id_to_label = {tid: lbl for lbl, tid in bundle.answer_token_ids.items()}
    answer_id_set = set(answer_id_to_label.keys())

    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            out = bundle.model(input_ids=ids, attention_mask=attn)
            logits = out.logits
            for i, ans_pos in enumerate(batch["answer_positions"]):
                pred_logits = logits[i, ans_pos - 1]
                mask = torch.full_like(pred_logits, float("-inf"))
                for tid in answer_id_set:
                    mask[tid] = 0.0
                pred_id = int(torch.argmax(pred_logits + mask).item())
                preds.append(answer_id_to_label[pred_id])
                golds.append(batch["gold_labels"][i])

    labels = sorted(set(golds) | set(preds))
    metrics = {
        "held_out_dataset": args.held_out_dataset,
        "n": int(len(golds)),
        "macro_f1": macro_f1(golds, preds, labels=labels),
        "per_class_f1": per_class_f1(golds, preds, labels=labels),
        "ci": f1_bootstrap_ci(golds, preds, n=1000, seed=args.seed),
    }
    out_path = Path(args.out or (Path(args.ckpt).parent / f"cross_{args.held_out_dataset}.json"))
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
