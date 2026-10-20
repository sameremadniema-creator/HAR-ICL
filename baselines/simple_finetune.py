"""Simple-finetune floor baseline (PDF Part 4)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from har_icl.utils.metrics import macro_f1, per_class_f1
from har_icl.utils.seeds import set_global_seed

from ._common import WindowsDataset, load_split_arrays, train_classifier
from .deepconvlstm import DeepConvLSTM


def _k_per_class(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keep_X, keep_y = [], []
    for lbl in sorted(set(y.tolist())):
        idx = np.where(y == lbl)[0]
        if len(idx) == 0:
            continue
        sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
        keep_X.append(X[sel]); keep_y.append(y[sel])
    return np.concatenate(keep_X, axis=0), np.concatenate(keep_y, axis=0)


def run(args) -> dict:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_split_arrays(Path(args.data), args.split)
    label_to_int = data["label_to_int"]
    int_to_label = {v: k for k, v in label_to_int.items()}

    # Build training set from K examples per class drawn from train + support_pool.
    train_X = data["train"]["X"]; train_y = data["train"]["y"]
    if "support_pool" in data:
        train_X = np.concatenate([train_X, data["support_pool"]["X"]], axis=0)
        train_y = np.concatenate([train_y, data["support_pool"]["y"]], axis=0)
    train_X, train_y = _k_per_class(train_X, train_y, args.k_shot, args.seed)
    y_int = np.array([label_to_int[str(l)] for l in train_y], dtype=np.int64)

    train_ds = WindowsDataset(train_X, y_int)
    val_ds = WindowsDataset(data["val"]["X"], data["val"]["y_int"])
    test_ds = WindowsDataset(data["test"]["X"], data["test"]["y_int"])

    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, max(1, len(train_ds))), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = DeepConvLSTM(
        in_channels=int(train_X.shape[-1]),
        num_classes=len(label_to_int),
    ).to(device)
    train_classifier(model, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            pred = logits.argmax(-1).cpu().numpy()
            preds.extend(int_to_label[int(p)] for p in pred)
            golds.extend(int_to_label[int(g)] for g in y.numpy())
    labels = sorted(set(golds) | set(preds))
    result = {
        "model": "SimpleFinetune (DeepConvLSTM)",
        "split": args.split, "seed": args.seed, "k_shot": args.k_shot,
        "n_test": int(len(golds)),
        "n_train": int(len(train_X)),
        "overall_macro_f1": macro_f1(golds, preds, labels=labels),
        "per_class_f1": per_class_f1(golds, preds, labels=labels),
    }
    out_path = Path(args.out or f"checkpoints/baselines/simple_finetune/split{args.split}_k{args.k_shot}_s{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/v1")
    p.add_argument("--split", type=str, default="C")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_shot", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
