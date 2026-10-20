"""Evaluate a trained GPT-2+LoRA checkpoint on a tokenised split."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from har_icl.phase3_gpt2.few_shot_dataset import FewShotICLDataset, collate_with_padding
from har_icl.phase3_gpt2.lora_setup import build_gpt2_with_lora
from har_icl.utils.metrics import macro_f1, per_class_f1
from har_icl.utils.seeds import set_global_seed

from .stats import f1_bootstrap_ci


def _restore_model(ckpt_dir: Path, num_codes: int, label_set, cfg: dict, device):
    bundle = build_gpt2_with_lora(
        num_codes=num_codes,
        activity_labels=label_set,
        rank=int(cfg["lora"]["rank"]),
        alpha=int(cfg["lora"]["alpha"]),
        dropout=float(cfg["lora"]["dropout"]),
        target_modules=cfg["lora"]["target_modules"],
        use_lora=False,
    )
    bundle.model.resize_token_embeddings(len(bundle.tokenizer))
    try:
        from peft import PeftModel  # type: ignore

        bundle.model = PeftModel.from_pretrained(bundle.model, ckpt_dir)
    except Exception as e:
        print(f"[eval] could not load LoRA adapter from {ckpt_dir}: {e}; using base GPT-2.")
    bundle.model.to(device).eval()
    return bundle


def evaluate(args) -> dict:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt)
    tokens_dir = Path(args.tokens)
    meta = json.loads((tokens_dir / "meta.json").read_text())
    label_set = list(meta["label_set"])
    num_codes = int(meta["num_codes"])
    # Pull the saved cfg from train summary if present.
    train_summary_path = ckpt_dir.parent / "summary.json"
    if train_summary_path.exists():
        cfg = json.loads(train_summary_path.read_text())["cfg"]
    else:
        cfg = {
            "model": {"context_length": 1024},
            "lora": {"rank": args.lora_rank, "alpha": 16, "dropout": 0.05, "target_modules": ["c_attn"]},
            "prompt": {"k_shot": args.k_shot, "max_distractors": 6},
            "train": {"batch_size": 16},
        }

    bundle = _restore_model(ckpt_dir, num_codes, label_set, cfg, device)
    pad_id = int(bundle.tokenizer.pad_token_id)
    collate = partial(collate_with_padding, pad_token_id=pad_id)
    test_ds = FewShotICLDataset(
        tokens_dir=tokens_dir, bundle=bundle, split=args.split,
        k_shot=int(args.k_shot),
        mode="test",
        max_distractors=int(cfg["prompt"]["max_distractors"]),
        max_length=int(cfg["model"]["context_length"]),
        seed=args.seed,
        # Must match how the model was trained (meta-ICL randomised labels).
        label_randomization=bool(cfg.get("prompt", {}).get("label_randomization", True)),
        distractor_mode=str(args.distractor_mode),
    )
    loader = DataLoader(test_ds, batch_size=int(cfg["train"]["batch_size"]),
                        shuffle=False, num_workers=0, collate_fn=collate)

    preds, golds = [], []
    cand_label_sets = []
    nll, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = bundle.model(input_ids=ids, attention_mask=attn, labels=labels)
            nll += float(out.loss) * ids.size(0)
            n += ids.size(0)
            logits = out.logits
            for i, ans_pos in enumerate(batch["answer_positions"]):
                pred_logits = logits[i, ans_pos - 1]
                # Per-episode decode: only this episode's candidate answer tokens are valid.
                cand = batch["cand_token_ids"][i]
                t2l = batch["tok2lbl"][i]
                mask = torch.full_like(pred_logits, float("-inf"))
                for tid in cand:
                    mask[tid] = 0.0
                pred_id = int(torch.argmax(pred_logits + mask).item())
                preds.append(t2l.get(pred_id, "<unk>"))
                golds.append(batch["gold_labels"][i])
                cand_label_sets.append([t2l.get(t, "<unk>") for t in cand])

    seen = set(test_ds.seen_labels)
    unseen = set(test_ds.unseen_labels)
    seen_pairs = [(g, p) for g, p in zip(golds, preds) if g in seen]
    unseen_pairs = [(g, p) for g, p in zip(golds, preds) if g in unseen]
    overall_labels = sorted(set(golds) | set(preds))
    metrics = {
        "split": args.split,
        "k_shot": int(args.k_shot),
        "n_test": int(n),
        "test_loss": nll / max(1, n),
        "overall_macro_f1": macro_f1(golds, preds, labels=overall_labels),
        # macro-F1 restricted to the SEEN / UNSEEN gold classes (labels = the gold set).
        "seen_macro_f1": (macro_f1([g for g, _ in seen_pairs], [p for _, p in seen_pairs],
                                   labels=sorted({g for g, _ in seen_pairs})) if seen_pairs else 0.0),
        "unseen_macro_f1": (macro_f1([g for g, _ in unseen_pairs], [p for _, p in unseen_pairs],
                                     labels=sorted({g for g, _ in unseen_pairs})) if unseen_pairs else 0.0),
        "seen_acc": (sum(1 for g, p in seen_pairs if g == p) / len(seen_pairs)) if seen_pairs else 0.0,
        "unseen_acc": (sum(1 for g, p in unseen_pairs if g == p) / len(unseen_pairs)) if unseen_pairs else 0.0,
        "unseen_per_class_f1": (per_class_f1([g for g, _ in unseen_pairs], [p for _, p in unseen_pairs],
                                             labels=sorted({g for g, _ in unseen_pairs})) if unseen_pairs else {}),
        "per_class_f1": per_class_f1(golds, preds, labels=overall_labels),
        "overall_ci": f1_bootstrap_ci(golds, preds, n=1000, seed=args.seed),
        "predictions": [{"gold": g, "pred": p} for g, p in zip(golds, preds)],
    }
    # Empirically-measured chance: a random classifier that picks uniformly among.
    try:
        import random as _rnd, statistics as _stats
        def _chance(target):
            idx = [j for j, g in enumerate(golds) if g in target and cand_label_sets[j]]
            if not idx:
                return None, None
            f1s, accs = [], []
            for r in range(50):
                rng = _rnd.Random(1000 + r)
                gg = [golds[j] for j in idx]
                pp = [rng.choice(cand_label_sets[j]) for j in idx]
                f1s.append(macro_f1(gg, pp, labels=sorted(set(gg))))
                accs.append(sum(1 for a, b in zip(gg, pp) if a == b) / len(gg))
            return round(_stats.mean(f1s), 4), round(_stats.mean(accs), 4)
        uf, ua = _chance(unseen)
        sf, sa = _chance(seen)
        metrics["chance"] = {"unseen_macro_f1": uf, "unseen_acc": ua,
                             "seen_macro_f1": sf, "seen_acc": sa, "reps": 50}
    except Exception as _e:  # noqa: BLE001
        metrics["chance"] = {"error": str(_e)}
    out_path = Path(args.out or (ckpt_dir.parent / "eval.json"))
    out_path.write_text(json.dumps(metrics, indent=2))
    # Trim predictions in the printed report.
    short = {k: v for k, v in metrics.items() if k != "predictions"}
    print(json.dumps(short, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to phase3 best checkpoint dir")
    parser.add_argument("--tokens", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distractor_mode", type=str, default="seen",
                        choices=["seen", "all_unseen", "mixed"],
                        help="'seen' (default, original protocol); 'all_unseen' = unseen-vs-unseen "
                             "N-way episodes; 'mixed' = seen+unseen distractors.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
