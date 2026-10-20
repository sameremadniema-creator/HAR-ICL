"""Chronos + MLP baseline (PDF Part 4)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from har_icl.utils.metrics import macro_f1, per_class_f1
from har_icl.utils.seeds import set_global_seed

from ._common import WindowsDataset, load_split_arrays, train_classifier


class FallbackEncoder(nn.Module):
    """Tiny trainable encoder used when chronos-forecasting is unavailable."""
    def __init__(self, in_channels: int = 6, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, out_dim, 5, padding=2), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).mean(dim=-1)


class ChronosMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 256, num_classes: int = 17) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, emb):
        return self.head(emb)


def _try_chronos_embed(X: np.ndarray, device) -> tuple[np.ndarray | None, int | None]:
    try:
        from chronos import ChronosPipeline  # type: ignore
    except Exception:
        return None, None
    try:
        pipe = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=str(device),
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"[chronos] failed to load pretrained: {e}")
        return None, None
    # Embed each channel separately, mean-pool across channels.
    B, T, C = X.shape
    embs = []
    for c in range(C):
        ctx = torch.from_numpy(X[:, :, c].astype(np.float32))
        # ChronosPipeline.embed expects a list of 1-D tensors.
        ems, _ = pipe.embed([ctx[i] for i in range(B)])
        # ems is a tensor (B, L, D) → mean over L.
        embs.append(ems.mean(dim=1).cpu().numpy())
    embed = np.stack(embs, axis=1).mean(axis=1)
    return embed.astype(np.float32), embed.shape[-1]


def run(args) -> dict:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_split_arrays(Path(args.data), args.split)
    int_to_label = {v: k for k, v in data["label_to_int"].items()}

    use_chronos = False
    embed_train = embed_val = embed_test = None
    embed_dim = None
    if args.try_chronos:
        embed_train, embed_dim = _try_chronos_embed(data["train"]["X"], device)
        if embed_train is not None:
            use_chronos = True
            embed_val, _ = _try_chronos_embed(data["val"]["X"], device)
            embed_test, _ = _try_chronos_embed(data["test"]["X"], device)

    if use_chronos:
        # Pre-computed continuous features; train a small MLP on top.
        head = ChronosMLP(embed_dim=embed_dim, num_classes=len(data["label_set"])).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        eb_t = torch.from_numpy(embed_train).to(device)
        y_t = torch.from_numpy(data["train"]["y_int"]).to(device)
        eb_v = torch.from_numpy(embed_val).to(device)
        y_v = torch.from_numpy(data["val"]["y_int"]).to(device)
        for ep in range(args.epochs):
            head.train()
            perm = torch.randperm(eb_t.size(0))
            for i in range(0, eb_t.size(0), args.batch_size):
                idx = perm[i : i + args.batch_size]
                logits = head(eb_t[idx])
                loss = crit(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            preds_t = head(torch.from_numpy(embed_test).to(device)).argmax(-1).cpu().numpy()
    else:
        encoder = FallbackEncoder(in_channels=int(data["train"]["X"].shape[-1])).to(device)
        head = ChronosMLP(embed_dim=64, num_classes=len(data["label_set"])).to(device)
        model = nn.Sequential(encoder, head)
        train_ds = WindowsDataset(data["train"]["X"], data["train"]["y_int"])
        val_ds = WindowsDataset(data["val"]["X"], data["val"]["y_int"])
        test_ds = WindowsDataset(data["test"]["X"], data["test"]["y_int"])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        train_classifier(model, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)
        model.eval()
        preds_t = []
        with torch.no_grad():
            for x, _ in test_loader:
                preds_t.append(model(x.to(device)).argmax(-1).cpu().numpy())
        preds_t = np.concatenate(preds_t)

    golds = [int_to_label[int(g)] for g in data["test"]["y_int"]]
    preds = [int_to_label[int(p)] for p in preds_t]
    labels = sorted(set(golds) | set(preds))
    result = {
        "model": "Chronos+MLP" if use_chronos else "Chronos+MLP (fallback encoder)",
        "split": args.split, "seed": args.seed,
        "n_test": int(len(golds)),
        "overall_macro_f1": macro_f1(golds, preds, labels=labels),
        "per_class_f1": per_class_f1(golds, preds, labels=labels),
        "used_chronos_pretrained": bool(use_chronos),
    }
    out_path = Path(args.out or f"checkpoints/baselines/chronos_mlp/split{args.split}_s{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/v1")
    p.add_argument("--split", type=str, default="C")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--try_chronos", action="store_true",
                   help="Try to load the chronos-t5-small pretrained encoder.")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
