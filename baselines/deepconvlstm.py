"""DeepConvLSTM (Ordóñez & Roggen, Sensors 2016, doi:10.3390/s16010115)."""
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


class DeepConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        conv_filters: int = 64,
        kernel_size: int = 5,
        lstm_hidden: int = 128,
        num_classes: int = 17,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_filters, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_filters, hidden_size=lstm_hidden, num_layers=2,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv(x)
        h = h.permute(0, 2, 1)  # (B, T, C)
        h, _ = self.lstm(h)
        return self.fc(h[:, -1])


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

    model = DeepConvLSTM(
        in_channels=int(data["train"]["X"].shape[-1]),
        num_classes=len(data["label_set"]),
    ).to(device)
    hist = train_classifier(model, train_loader, val_loader, device=device,
                            epochs=args.epochs, lr=args.lr, weight_decay=1e-4)

    # Test eval (seen-only — DeepConvLSTM cannot predict unseen classes by design).
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(-1).cpu().numpy()
            preds.extend(int_to_label[int(p)] for p in pred)
            golds.extend(int_to_label[int(g)] for g in y.numpy())
    seen = set(int_to_label.values())
    # By construction the DeepConvLSTM has a fixed classifier head; unseen-class
    overall_labels = sorted(set(golds) | set(preds))
    seen_mask = [g in seen for g in golds]
    result = {
        "model": "DeepConvLSTM",
        "split": args.split,
        "seed": args.seed,
        "n_test": int(len(golds)),
        "overall_macro_f1": macro_f1(golds, preds, labels=overall_labels),
        "per_class_f1": per_class_f1(golds, preds, labels=overall_labels),
        "history_tail": hist["history"][-3:],
    }
    out_path = Path(args.out or f"checkpoints/baselines/deepconvlstm/split{args.split}_s{args.seed}.json")
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
