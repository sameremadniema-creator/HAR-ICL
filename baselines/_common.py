"""Shared helpers for baseline trainers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class WindowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y_int: np.ndarray) -> None:
        self.X = X.astype(np.float32)
        self.y = y_int.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        # (T, C) → (C, T) for Conv1d
        x = torch.from_numpy(self.X[i]).transpose(0, 1).contiguous()
        return x, int(self.y[i])


def load_split_arrays(
    data_dir: Path,
    split: str,
    label_to_int: dict[str, int] | None = None,
) -> dict:
    """Read data/processed/v1/{windows.npz,splits.json,meta.json}, return per-subset arrays."""
    data = np.load(data_dir / "windows.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(object)
    meta = json.loads((data_dir / "meta.json").read_text())
    if label_to_int is None:
        label_to_int = meta["label_to_int"]
    splits = json.loads((data_dir / "splits.json").read_text())
    per_window = np.array(splits["splits"][split]["per_window"], dtype=object)
    out: dict = {"label_set": meta["label_set"], "label_to_int": label_to_int, "meta": meta}
    for subset in ("train", "val", "test", "support_pool"):
        mask = per_window == subset
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        y_sub = y[idx]
        y_int = np.array([label_to_int[str(lbl)] for lbl in y_sub], dtype=np.int64)
        out[subset] = {"X": X[idx], "y": y_sub, "y_int": y_int, "idx": idx}
    return out


def macro_f1_from_logits(logits: torch.Tensor, y: np.ndarray, int_to_label: dict[int, str]) -> tuple[float, list[str], list[str]]:
    pred = logits.argmax(dim=-1).cpu().numpy()
    preds = [int_to_label[int(i)] for i in pred]
    golds = [str(g) for g in y]
    from har_icl.utils.metrics import macro_f1

    labels = sorted(set(golds) | set(preds))
    return macro_f1(golds, preds, labels=labels), preds, golds


def train_classifier(
    model,
    train_loader,
    val_loader,
    *,
    device,
    epochs: int = 30,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
) -> dict:
    import torch.nn as nn
    import torch.optim as optim

    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    history = []
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            tl += float(loss) * x.size(0); n += x.size(0)
        # val.
        model.eval()
        vl, vn, correct = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = crit(logits, y)
                vl += float(loss) * x.size(0); vn += x.size(0)
                correct += int((logits.argmax(-1) == y).sum())
        history.append({
            "epoch": ep, "train_loss": tl / max(1, n),
            "val_loss": vl / max(1, vn), "val_acc": correct / max(1, vn),
        })
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best_val}
