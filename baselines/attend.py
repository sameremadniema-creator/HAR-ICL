"""Attend (Abedin et al., IMWUT 2021) — BiLSTM + temporal attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from har_icl.utils.metrics import macro_f1, per_class_f1
from har_icl.utils.seeds import set_global_seed

from ._common import WindowsDataset, load_split_arrays, train_classifier


class Attend(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        attn_hidden: int = 128,
        num_classes: int = 17,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=in_channels, hidden_size=bilstm_hidden, num_layers=bilstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        d_out = bilstm_hidden * 2
        self.attn = nn.Sequential(
            nn.Linear(d_out, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1, bias=False),
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_out, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C) for LSTM.
        h = x.permute(0, 2, 1)
        h, _ = self.bilstm(h)
        # Temporal attention.
        scores = self.attn(h).squeeze(-1)   # (B, T)
        w = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * w).sum(dim=1)
        return self.fc(pooled)


def run(args) -> dict:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_split_arrays(Path(args.data), args.split)
    int_to_label = {v: k for k, v in data["label_to_int"].items()}

    train_ds = WindowsDataset(data["train"]["X"], data["train"]["y_int"])
    val_ds = WindowsDataset(data["val"]["X"], data["val"]["y_int"])
    test_ds = WindowsDataset(data["test"]["X"], data["test"]["y_int"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = Attend(
        in_channels=int(data["train"]["X"].shape[-1]),
        num_classes=len(data["label_set"]),
    ).to(device)
    hist = train_classifier(model, train_loader, val_loader, device=device, epochs=args.epochs, lr=args.lr)

    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            pred = logits.argmax(-1).cpu().numpy()
            preds.extend(int_to_label[int(p)] for p in pred)
            golds.extend(int_to_label[int(g)] for g in y.numpy())
    overall_labels = sorted(set(golds) | set(preds))
    result = {
        "model": "Attend",
        "split": args.split, "seed": args.seed,
        "n_test": int(len(golds)),
        "overall_macro_f1": macro_f1(golds, preds, labels=overall_labels),
        "per_class_f1": per_class_f1(golds, preds, labels=overall_labels),
        "history_tail": hist["history"][-3:],
    }
    out_path = Path(args.out or f"checkpoints/baselines/attend/split{args.split}_s{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/v1")
    p.add_argument("--split", type=str, default="C")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
