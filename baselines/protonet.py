"""Prototypical Networks (Snell et al."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from har_icl.utils.metrics import macro_f1, per_class_f1
from har_icl.utils.seeds import set_global_seed

from ._common import load_split_arrays


class ProtoEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 64, latent: int = 64) -> None:
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, 5, padding=2), nn.BatchNorm1d(c_out), nn.ReLU(), nn.MaxPool1d(2),
            )
        self.net = nn.Sequential(
            block(in_channels, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
            block(hidden, latent),
        )

    def forward(self, x):
        h = self.net(x)
        return h.mean(dim=-1)


def _sample_episode(
    rng: np.random.Generator,
    pool_X: np.ndarray, pool_y: np.ndarray,
    n_way: int, k_shot: int, q_query: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    labels = sorted(set(pool_y.tolist()))
    if len(labels) < n_way:
        n_way = len(labels)
    chosen = list(rng.choice(labels, size=n_way, replace=False))
    sup_X, sup_y, q_X, q_y = [], [], [], []
    for ci, lbl in enumerate(chosen):
        idx = np.where(pool_y == lbl)[0]
        if len(idx) < k_shot + q_query:
            idx = rng.choice(idx, size=k_shot + q_query, replace=True)
        else:
            idx = rng.choice(idx, size=k_shot + q_query, replace=False)
        sup_X.append(pool_X[idx[:k_shot]])
        sup_y.extend([ci] * k_shot)
        q_X.append(pool_X[idx[k_shot:]])
        q_y.extend([ci] * q_query)
    sup_X = np.concatenate(sup_X, axis=0).astype(np.float32)
    q_X = np.concatenate(q_X, axis=0).astype(np.float32)
    return (
        torch.from_numpy(sup_X).transpose(1, 2).contiguous(),
        torch.tensor(sup_y, dtype=torch.long),
        torch.from_numpy(q_X).transpose(1, 2).contiguous(),
        q_y,
    )


def _proto_loss_and_acc(emb_sup, y_sup, emb_q, y_q, n_way):
    proto = torch.stack([emb_sup[y_sup == c].mean(0) for c in range(n_way)], dim=0)
    # Negative squared distance to each prototype = logits.
    logits = -torch.cdist(emb_q, proto) ** 2
    loss = F.cross_entropy(logits, y_q)
    acc = float((logits.argmax(-1) == y_q).float().mean())
    return loss, acc


def run(args) -> dict:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_split_arrays(Path(args.data), args.split)
    rng = np.random.default_rng(args.seed)

    encoder = ProtoEncoder(in_channels=int(data["train"]["X"].shape[-1])).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr)

    train_X = data["train"]["X"]
    train_y = data["train"]["y"]
    # Training episodes use seen classes only.
    n_way = min(args.n_way, len(set(train_y.tolist())))
    losses = []
    for ep in range(args.episodes):
        sup_X, sup_y, q_X, q_y = _sample_episode(rng, train_X, train_y, n_way, args.k_shot, args.q_query)
        sup_X = sup_X.to(device); q_X = q_X.to(device)
        sup_y = sup_y.to(device); q_y = torch.tensor(q_y, device=device)
        emb_sup = encoder(sup_X)
        emb_q = encoder(q_X)
        loss, acc = _proto_loss_and_acc(emb_sup, sup_y, emb_q, q_y, n_way)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append({"episode": ep, "loss": float(loss), "acc": acc})

    # Evaluation: for each test window, K-shot prototypes from train+support_pool
    test = data["test"]
    pool_X = data["train"]["X"]
    pool_y = data["train"]["y"]
    pool2 = data.get("support_pool")
    if pool2 is not None:
        pool_X = np.concatenate([pool_X, pool2["X"]], axis=0)
        pool_y = np.concatenate([pool_y, pool2["y"]], axis=0)

    encoder.eval()
    preds, golds = [], []
    with torch.no_grad():
        # Compute one prototype per label by averaging args.k_shot random support.
        unique = sorted(set(pool_y.tolist()))
        proto_labels = []
        proto_vecs = []
        for lbl in unique:
            idx = np.where(pool_y == lbl)[0]
            if len(idx) == 0:
                continue
            k = min(args.k_shot, len(idx))
            sel = rng.choice(idx, size=k, replace=False)
            xb = torch.from_numpy(pool_X[sel].astype(np.float32)).transpose(1, 2).contiguous().to(device)
            proto = encoder(xb).mean(0)
            proto_labels.append(str(lbl))
            proto_vecs.append(proto)
        proto_mat = torch.stack(proto_vecs, dim=0)

        for i in range(0, len(test["X"]), args.batch_size):
            xb = torch.from_numpy(test["X"][i : i + args.batch_size].astype(np.float32)).transpose(1, 2).contiguous().to(device)
            emb = encoder(xb)
            dist = torch.cdist(emb, proto_mat)
            pred_idx = dist.argmin(-1).cpu().numpy()
            preds.extend(proto_labels[int(p)] for p in pred_idx)
            golds.extend(str(g) for g in test["y"][i : i + args.batch_size])

    overall_labels = sorted(set(golds) | set(preds))
    result = {
        "model": "ProtoNet",
        "split": args.split, "seed": args.seed,
        "k_shot": args.k_shot, "episodes": args.episodes,
        "overall_macro_f1": macro_f1(golds, preds, labels=overall_labels),
        "per_class_f1": per_class_f1(golds, preds, labels=overall_labels),
        "loss_tail": losses[-5:],
    }
    out_path = Path(args.out or f"checkpoints/baselines/protonet/split{args.split}_k{args.k_shot}_s{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/v1")
    p.add_argument("--split", type=str, default="C")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=4)
    p.add_argument("--q_query", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
