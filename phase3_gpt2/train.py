"""Phase 3 trainer: GPT-2 + LoRA on K-shot ICL prompts."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from har_icl.utils.checkpointing import GracefulCheckpoint, save_torch_checkpoint
from har_icl.utils.logging import JSONLLogger, init_wandb
from har_icl.utils.manifest import write_manifest
from har_icl.utils.metrics import macro_f1, per_class_f1, overfit_report
from har_icl.utils.seeds import set_global_seed

from .few_shot_dataset import FewShotICLDataset, collate_with_padding
from .lora_setup import build_gpt2_with_lora, trainable_parameter_count


def _resolve_config(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    base = cfg.get("inherit")
    if base:
        base_path = cfg_path.parent / base
        base_cfg = yaml.safe_load(base_path.read_text())
        merged = _deep_merge(base_cfg, cfg)
        merged.pop("inherit", None)
        return merged
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _eval(model, loader, device) -> dict:
    model.eval()
    preds: list[str] = []
    golds: list[str] = []
    nll = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            nll += float(out.loss) * ids.size(0)
            n += ids.size(0)
            logits = out.logits  # (B, L, V)
            # Per-episode decode: restrict to THIS episode's candidate answer tokens
            for i, ans_pos in enumerate(batch["answer_positions"]):
                pred_logits = logits[i, ans_pos - 1]
                cand = batch["cand_token_ids"][i]
                t2l = batch["tok2lbl"][i]
                mask = torch.full_like(pred_logits, float("-inf"))
                for tid in cand:
                    mask[tid] = 0.0
                pred_id = int(torch.argmax(pred_logits + mask).item())
                preds.append(t2l.get(pred_id, "<unk>"))
                golds.append(batch["gold_labels"][i])
    if not golds:
        return {"loss": 0.0, "macro_f1": 0.0, "per_class_f1": {}, "n": 0, "preds": [], "golds": []}
    label_universe = sorted(set(golds) | set(preds))
    f1m = macro_f1(golds, preds, labels=label_universe)
    f1pc = per_class_f1(golds, preds, labels=label_universe)
    return {
        "loss": nll / max(1, n),
        "macro_f1": f1m,
        "per_class_f1": f1pc,
        "n": int(n),
        "preds": preds,
        "golds": golds,
    }


def train(args) -> dict:
    set_global_seed(args.seed)
    cfg_path = Path(args.config)
    cfg = _resolve_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens_dir = Path(args.tokens)
    meta = json.loads((tokens_dir / "meta.json").read_text())
    label_set = list(meta["label_set"])
    num_codes = int(meta["num_codes"])

    bundle = build_gpt2_with_lora(
        num_codes=num_codes,
        activity_labels=label_set,
        rank=int(args.lora_rank or cfg["lora"]["rank"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        target_modules=cfg["lora"]["target_modules"],
    )
    trainable, total = trainable_parameter_count(bundle.model)
    print(
        f"[gpt2-lora] trainable {trainable:,} / {total:,} "
        f"({100.0 * trainable / max(1, total):.3f}%)"
    )

    bundle.model.to(device)
    answer_id_to_label = {tid: lbl for lbl, tid in bundle.answer_token_ids.items()}
    answer_id_set = set(answer_id_to_label.keys())
    pad_id = int(bundle.tokenizer.pad_token_id)
    collate = partial(collate_with_padding, pad_token_id=pad_id)

    max_train = args.max_examples if args.max_examples and args.max_examples > 0 else None
    max_eval = args.max_eval_examples if args.max_eval_examples and args.max_eval_examples > 0 else None
    # Meta-ICL: randomise class->answer-token per episode (default on) so the model
    label_rand = bool(cfg["prompt"].get("label_randomization", True))
    train_ds = FewShotICLDataset(
        tokens_dir=tokens_dir, bundle=bundle, split=args.split,
        k_shot=int(args.k_shot or cfg["prompt"]["k_shot"]),
        mode="train", max_distractors=int(cfg["prompt"]["max_distractors"]),
        max_length=int(cfg["model"]["context_length"]),
        seed=args.seed,
        max_examples=max_train,
        label_randomization=label_rand,
    )
    val_ds = FewShotICLDataset(
        tokens_dir=tokens_dir, bundle=bundle, split=args.split,
        k_shot=int(args.k_shot or cfg["prompt"]["k_shot"]),
        mode="val", max_distractors=int(cfg["prompt"]["max_distractors"]),
        max_length=int(cfg["model"]["context_length"]),
        seed=args.seed,
        max_examples=max_eval,
        label_randomization=label_rand,
    )
    test_ds = FewShotICLDataset(
        tokens_dir=tokens_dir, bundle=bundle, split=args.split,
        k_shot=int(args.k_shot or cfg["prompt"]["k_shot"]),
        mode="test", max_distractors=int(cfg["prompt"]["max_distractors"]),
        max_length=int(cfg["model"]["context_length"]),
        seed=args.seed,
        max_examples=max_eval,
        label_randomization=label_rand,
    )

    bsz = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False, num_workers=0, collate_fn=collate)

    optimiser = optim.AdamW(
        [p for p in bundle.model.parameters() if p.requires_grad],
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    epochs = int(args.epochs or cfg["train"]["epochs"])
    # Gradient accumulation: micro-batch = cfg batch_size, optimiser steps every
    accum = max(1, int(cfg["train"].get("grad_accum", 1)))
    grad_clip = float(cfg["train"]["grad_clip"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(out_dir / "train.jsonl")
    init_wandb(project="har_icl", name=f"gpt2_{cfg['name']}_split{args.split}_s{args.seed}", config=cfg, offline=True)
    grace = GracefulCheckpoint()

    best_f1 = -1.0
    best_ep = -1
    patience = int(getattr(args, "patience", 0) or 0)
    best_path = out_dir / "best"
    best_path.mkdir(exist_ok=True)
    history = []

    for epoch in range(epochs):
        bundle.model.train()
        ep_loss, n = 0.0, 0
        optimiser.zero_grad(set_to_none=True)
        n_batches = len(train_loader)
        for i, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = bundle.model(input_ids=ids, attention_mask=attn, labels=labels)
            # Scale by accum so summed micro-batch grads == mean over effective batch.
            (out.loss / accum).backward()
            # Step at every `accum`-th micro-batch, and flush the final remainder.
            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                nn.utils.clip_grad_norm_(bundle.model.parameters(), grad_clip)
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            ep_loss += float(out.loss) * ids.size(0)
            n += ids.size(0)

        val_metrics = _eval(bundle.model, val_loader, device)
        # Split val performance further by seen/unseen classes (informational).
        seen = set(train_ds.seen_labels)
        unseen_golds = [g for g in val_metrics["golds"] if g not in seen]
        unseen_preds = [p for g, p in zip(val_metrics["golds"], val_metrics["preds"]) if g not in seen]
        seen_golds = [g for g in val_metrics["golds"] if g in seen]
        seen_preds = [p for g, p in zip(val_metrics["golds"], val_metrics["preds"]) if g in seen]
        seen_f1 = macro_f1(seen_golds, seen_preds, labels=sorted(set(seen_golds))) if seen_golds else 0.0
        unseen_f1 = macro_f1(unseen_golds, unseen_preds, labels=sorted(set(unseen_golds))) if unseen_golds else 0.0
        blended = 0.5 * (seen_f1 + unseen_f1)
        record = {
            "epoch": int(epoch),
            "train_loss": ep_loss / max(1, n),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_seen_f1": seen_f1,
            "val_unseen_f1": unseen_f1,
            "val_blended_f1": blended,
            "trainable_params": int(trainable),
        }
        logger.log(**record)
        history.append(record)
        print(
            f"[ep {epoch+1}/{epochs}] tr_loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} "
            f"seen_F1={seen_f1:.3f} unseen_F1={unseen_f1:.3f} blended={blended:.3f}"
        )

        if blended > best_f1:
            best_f1 = blended
            best_ep = epoch
            # Persist best state.
            bundle.model.save_pretrained(best_path)
            bundle.tokenizer.save_pretrained(best_path)
            (best_path / "metrics.json").write_text(json.dumps(record, indent=2))

        grace.check(lambda: bundle.model.save_pretrained(out_dir / "last"))

        # Early stopping (matched to the continuous control).
        if patience > 0 and (epoch - best_ep) >= patience:
            print(f"[early-stop] no val_blended improvement for {patience} epochs "
                  f"(best={best_f1:.4f} @ ep{best_ep}); stopping at ep{epoch}.", flush=True)
            break

    overfit_report(history)

    # Final test eval using best checkpoint.
    from transformers import GPT2LMHeadModel

    try:
        # Reload best (peft-aware).
        from peft import PeftModel  # type: ignore

        # Rebuild base + load peft adapter.
        base_bundle = build_gpt2_with_lora(
            num_codes=num_codes, activity_labels=label_set,
            rank=int(args.lora_rank or cfg["lora"]["rank"]),
            alpha=int(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
            target_modules=cfg["lora"]["target_modules"],
            use_lora=False,
        )
        base_bundle.model.resize_token_embeddings(len(bundle.tokenizer))
        try:
            test_model = PeftModel.from_pretrained(base_bundle.model, best_path).to(device)
        except Exception:
            test_model = bundle.model
    except ImportError:
        test_model = bundle.model
    test_metrics = _eval(test_model, test_loader, device)

    seen_golds = [g for g in test_metrics["golds"] if g in train_ds.seen_labels]
    seen_preds = [p for g, p in zip(test_metrics["golds"], test_metrics["preds"]) if g in train_ds.seen_labels]
    unseen_golds = [g for g in test_metrics["golds"] if g in train_ds.unseen_labels]
    unseen_preds = [p for g, p in zip(test_metrics["golds"], test_metrics["preds"]) if g in train_ds.unseen_labels]
    # labels= the gold set, so unseen-F1 is over UNSEEN classes only (matches.
    test_seen_f1 = macro_f1(seen_golds, seen_preds, labels=sorted(set(seen_golds))) if seen_golds else 0.0
    test_unseen_f1 = macro_f1(unseen_golds, unseen_preds, labels=sorted(set(unseen_golds))) if unseen_golds else 0.0
    test_unseen_acc = (sum(1 for g, p in zip(unseen_golds, unseen_preds) if g == p) / len(unseen_golds)
                       if unseen_golds else 0.0)
    summary = {
        "best_val_blended_f1": float(best_f1),
        "test_macro_f1": test_metrics["macro_f1"],
        "test_seen_f1": test_seen_f1,
        "test_unseen_f1": test_unseen_f1,
        "test_unseen_acc": test_unseen_acc,
        "test_per_class_f1": test_metrics["per_class_f1"],
        "test_n": test_metrics["n"],
        "history": history,
        "cfg": cfg,
        "trainable_params": int(trainable),
        "total_params": int(total),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_manifest(
        out_dir=Path("manifests"),
        run_id=f"gpt2_{cfg['name']}_split{args.split}_s{args.seed}",
        config=cfg,
        ckpt_path=best_path / "adapter_model.safetensors" if (best_path / "adapter_model.safetensors").exists() else None,
        metrics={
            "test_macro_f1": test_metrics["macro_f1"],
            "test_seen_f1": test_seen_f1,
            "test_unseen_f1": test_unseen_f1,
            "best_val_blended_f1": float(best_f1),
        },
        extra={"phase": "gpt2", "split": args.split, "seed": args.seed, "k_shot": args.k_shot},
    )
    logger.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2/_base.yaml")
    parser.add_argument("--tokens", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--k_shot", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stop patience on val_blended_f1 (0 = off). Set equal to the "
                             "continuous control's patience for a fully matched discrete-vs-continuous run.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=0,
                        help="Cap training examples per epoch (0 = use all). Smoke-test knob.")
    parser.add_argument("--max_eval_examples", type=int, default=0,
                        help="Cap eval examples (0 = use all). Smoke-test knob.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
