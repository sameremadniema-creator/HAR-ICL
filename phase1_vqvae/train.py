"""Phase 1 trainer: VQ-VAE on harmonised IMU windows."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from har_icl.utils.checkpointing import GracefulCheckpoint, save_torch_checkpoint
from har_icl.utils.logging import JSONLLogger, init_wandb
from har_icl.utils.manifest import hash_obj, write_manifest
from har_icl.utils.metrics import codebook_usage, overfit_report
from har_icl.utils.seeds import set_global_seed

from .conditioning import TextEmbeddingCache, build_label_to_text, fallback_random_embedding
from .model import ActivityAwareVQVAE


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


class WindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        text_embs: dict[str, np.ndarray] | None,
        label_to_int: dict[str, int],
        use_conditioning: bool,
        text_dim: int,
        augment: bool = False,
        aug_jitter: float = 0.05,
        aug_scale: float = 0.1,
        aug_shift: int = 8,
        ssl: bool = False,
        aug_rot_deg: float = 25.0,
        aug_timewarp: float = 0.2,
        aug_sensor_drop: float = 0.15,
    ) -> None:
        self.X = X.astype(np.float32)
        self.y = np.array(y, dtype=object)
        self.text_embs = text_embs
        self.label_to_int = label_to_int
        self.use_conditioning = use_conditioning
        self.text_dim = text_dim
        # IMU augmentation (train only): regularises the codebook so it generalises
        self.augment = bool(augment)
        self.aug_jitter = float(aug_jitter)
        self.aug_scale = float(aug_scale)
        self.aug_shift = int(aug_shift)
        # SSL two-view mode: produce a SECOND, strongly-augmented view per window for
        self.ssl = bool(ssl)
        self.aug_rot_deg = float(aug_rot_deg)
        self.aug_timewarp = float(aug_timewarp)
        self.aug_sensor_drop = float(aug_sensor_drop)

    def __len__(self) -> int:
        return len(self.X)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T) — mild aug used on the reconstruction view.
        C = x.shape[0]
        if self.aug_scale > 0:                       # per-channel amplitude scaling
            x = x * (1.0 + torch.randn(C, 1) * self.aug_scale)
        if self.aug_jitter > 0:                      # additive sensor noise
            x = x + torch.randn_like(x) * self.aug_jitter
        if self.aug_shift > 0:                       # small temporal shift
            s = int(torch.randint(-self.aug_shift, self.aug_shift + 1, (1,)).item())
            x = torch.roll(x, shifts=s, dims=1)
        return x

    def _rand_rotation(self, max_deg: float) -> torch.Tensor:
        """Random BOUNDED rotation matrix (small angles on all axes => gravity-safe)."""
        th = np.radians(np.random.uniform(-max_deg, max_deg, size=3))
        cx, cy, cz = np.cos(th)
        sx, sy, sz = np.sin(th)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return torch.from_numpy((Rz @ Ry @ Rx).astype(np.float32))

    def _time_warp(self, x: torch.Tensor, sigma: float, knots: int = 4) -> torch.Tensor:
        C, T = x.shape
        orig = np.linspace(0, T - 1, T)
        steps = np.linspace(0, T - 1, knots + 2)
        rand = np.random.normal(1.0, sigma, knots + 2).clip(0.2, None)
        cum = np.cumsum(rand)
        cum = (cum - cum.min()) / (cum.max() - cum.min() + 1e-8) * (T - 1)
        new_idx = np.interp(orig, steps, cum)
        xn = x.numpy()
        out = np.stack([np.interp(orig, new_idx, xn[c]) for c in range(C)]).astype(np.float32)
        return torch.from_numpy(out)

    def _augment_ssl(self, x: torch.Tensor) -> torch.Tensor:
        """Strong, gravity-safe augmentation for the SSL contrastive view."""
        if self.aug_rot_deg > 0 and x.shape[0] >= 6:
            R = self._rand_rotation(self.aug_rot_deg)
            x = x.clone()
            x[0:3] = R @ x[0:3]
            x[3:6] = R @ x[3:6]
        if self.aug_timewarp > 0 and np.random.rand() < 0.8:
            x = self._time_warp(x, self.aug_timewarp)
        if self.aug_sensor_drop > 0 and x.shape[0] >= 6:  # drop a whole accel/gyro triad
            x = x.clone()
            if np.random.rand() < self.aug_sensor_drop:
                x[0:3] = 0.0
            if np.random.rand() < self.aug_sensor_drop:
                x[3:6] = 0.0
        x = x * (1.0 + torch.randn(x.shape[0], 1) * self.aug_scale)
        x = x + torch.randn_like(x) * self.aug_jitter
        return x

    def __getitem__(self, idx: int):
        # (T, C) -> (C, T) for Conv1d
        x0 = torch.from_numpy(self.X[idx]).transpose(0, 1).contiguous()
        x = self._augment(x0) if self.augment else x0
        # Second view for SSL (strong aug); identity copy when SSL off (unused.
        x2 = self._augment_ssl(x0) if self.ssl else x
        lbl = self.y[idx]
        label_int = int(self.label_to_int[lbl])
        if self.use_conditioning and self.text_embs is not None and lbl in self.text_embs:
            text_emb = torch.from_numpy(self.text_embs[lbl].astype(np.float32))
        else:
            text_emb = torch.zeros(self.text_dim, dtype=torch.float32)
        return x, x2, label_int, text_emb


def _load_data(data_dir: Path, split: str = "C") -> dict:
    data = np.load(data_dir / "windows.npz", allow_pickle=True)
    X = data["X"]                                  # (N, T, C)
    y = data["y"].astype(object)
    subjects = data["subjects"].astype(object)
    datasets = data["datasets"].astype(object)
    meta = json.loads((data_dir / "meta.json").read_text())
    splits = json.loads((data_dir / "splits.json").read_text())
    per_window = np.array(splits["splits"][split]["per_window"], dtype=object)
    return {
        "X": X, "y": y, "subjects": subjects, "datasets": datasets,
        "per_window": per_window, "meta": meta,
        "label_set": meta["label_set"], "label_to_int": meta["label_to_int"],
        "label_descriptions": meta["label_descriptions"],
    }


def _make_loaders(data: dict, cfg: dict, text_embs: dict | None) -> tuple[DataLoader, DataLoader]:
    per_window = data["per_window"]
    train_idx = np.where(per_window == "train")[0]
    val_idx = np.where(per_window == "val")[0]

    use_cond = bool(cfg["conditioning"]["enabled"])
    text_dim = int(cfg["conditioning"]["text_dim"])

    tr = cfg.get("train", {})
    aug = bool(tr.get("augment", False))
    ssl = float(cfg["loss"].get("ssl_contrastive_weight", 0.0)) > 0.0
    train_ds = WindowDataset(
        data["X"][train_idx], data["y"][train_idx], text_embs,
        data["label_to_int"], use_cond, text_dim,
        augment=aug,
        aug_jitter=float(tr.get("aug_jitter", 0.05)),
        aug_scale=float(tr.get("aug_scale", 0.1)),
        aug_shift=int(tr.get("aug_shift", 8)),
        ssl=ssl,
        aug_rot_deg=float(tr.get("aug_rot_deg", 25.0)),
        aug_timewarp=float(tr.get("aug_timewarp", 0.2)),
        aug_sensor_drop=float(tr.get("aug_sensor_drop", 0.15)),
    )
    val_ds = WindowDataset(
        data["X"][val_idx], data["y"][val_idx], text_embs,
        data["label_to_int"], use_cond, text_dim,
        augment=False,   # never augment validation
    )
    bsz = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, num_workers=0)
    return train_loader, val_loader, val_idx


def _transfer_val_f1(
    model,
    X_val: np.ndarray,
    y_val_int: np.ndarray,
    holdout_ids: list[int],
    device,
    k_shot: int = 4,
    n_distractors: int = 6,
    reps: int = 3,
    max_queries: int = 100,
    seed: int = 1234,
    batch_size: int = 256,
) -> float:
    """Transfer-aware validation metric: euclidean nearest-prototype F1 on the held-out"""
    import random as _random
    model.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(X_val), batch_size):
            xb = torch.from_numpy(X_val[s:s + batch_size].astype(np.float32)).transpose(1, 2).contiguous().to(device)
            feats.append(model.encoder(xb).mean(dim=-1).cpu().numpy())
    F = np.concatenate(feats, axis=0)
    by_cls: dict[int, list[int]] = {}
    for i, lab in enumerate(y_val_int):
        by_cls.setdefault(int(lab), []).append(i)
    eligible = [c for c, idx in by_cls.items() if len(idx) >= k_shot + 1]
    golds, preds = [], []
    for r in range(reps):
        rng = _random.Random(seed + r)
        for h in holdout_ids:
            if h not in by_cls or len(by_cls[h]) < k_shot + 1:
                continue
            others = [c for c in eligible if c != h]
            rng.shuffle(others)
            cands = others[:n_distractors] + [h]
            protos, plabels = [], []
            h_pool = by_cls[h][:]
            rng.shuffle(h_pool)
            support, queries = h_pool[:k_shot], h_pool[k_shot:k_shot + max_queries]
            for c in cands:
                if c == h:
                    protos.append(F[support].mean(axis=0))
                else:
                    picks = [rng.choice(by_cls[c]) for _ in range(k_shot)]
                    protos.append(F[picks].mean(axis=0))
                plabels.append(c)
            P = np.stack(protos, axis=0)
            for qi in queries:
                d = np.linalg.norm(P - F[qi][None, :], axis=1)
                golds.append(h)
                preds.append(plabels[int(d.argmin())])
    if not golds:
        return 0.0
    from har_icl.utils.metrics import macro_f1
    return float(macro_f1([str(g) for g in golds], [str(p) for p in preds]))


def _cosine_lr(step: int, total: int, base: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * p))


def _label_text_map(label_set, label_descriptions) -> dict[str, str]:
    return {lbl: label_descriptions.get(lbl, lbl) for lbl in label_set}


def _get_text_embeddings(cfg, label_set, label_descriptions, cache_path):
    if not cfg["conditioning"]["enabled"]:
        return None
    label_to_text = _label_text_map(label_set, label_descriptions)
    try:
        cache = TextEmbeddingCache(
            model_name=cfg["conditioning"]["text_model"],
            cache_path=cache_path,
            device="cpu",
        )
        text_embs = cache.encode_dict(label_to_text)
        print(f"[text-cache] cached {len(text_embs)} labels at {cache_path}")
        return text_embs
    except Exception as e:
        print(f"[text-cache] sentence-transformers unavailable ({e}); using fallback random embeddings.")
        return fallback_random_embedding(label_set, dim=int(cfg["conditioning"]["text_dim"]), seed=0)


def train(args) -> dict:
    set_global_seed(args.seed)
    cfg_path = Path(args.config)
    cfg = _resolve_config(cfg_path)

    # batch_size overrides the config (0 = use config) to fit a shared/contended 3090.
    if int(getattr(args, "batch_size", 0) or 0) > 0:
        cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
        print(f"[train_vqvae] batch_size overridden to {args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_vqvae] config={cfg['name']} seed={args.seed} device={device}")

    data = _load_data(Path(args.data), split=args.split)
    label_set = data["label_set"]
    num_aux_classes = len(label_set)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    text_cache_path = ckpt_dir / "text_emb.npz"

    text_embs = _get_text_embeddings(cfg, label_set, data["label_descriptions"], text_cache_path)

    train_loader, val_loader, val_idx = _make_loaders(data, cfg, text_embs)

    model = ActivityAwareVQVAE(
        in_channels=int(cfg["encoder"]["in_channels"]),
        hidden_channels=list(cfg["encoder"]["hidden_channels"]),
        latent_dim=int(cfg["encoder"]["latent_dim"]),
        num_codes=int(cfg["codebook"]["num_codes"]),
        codebook_decay=float(cfg["codebook"]["decay"]),
        commit_beta=float(cfg["loss"]["commit_beta"]),
        use_conditioning=bool(cfg["conditioning"]["enabled"]),
        text_dim=int(cfg["conditioning"]["text_dim"]),
        use_aux_classifier=bool(cfg["aux_classifier"]["enabled"]),
        num_aux_classes=num_aux_classes,
        recon_l1_weight=float(cfg["loss"]["recon_l1_weight"]),
        aux_ce_weight=float(cfg["loss"]["aux_ce_weight"]),
        kernel_size=int(cfg["encoder"].get("kernel_size", 5)),
        encoder_strides=cfg["encoder"].get("strides"),                       # e.g. [2,2,1] -> 32 tokens
        contrastive_weight=float(cfg["loss"].get("contrastive_weight", 0.0)),
        contrastive_temperature=float(cfg["loss"].get("contrastive_temperature", 0.1)),
        quantizer_type=str(cfg["codebook"].get("type", "vq")),               # "vq" (EMA) or "fsq"
        fsq_levels=cfg["codebook"].get("levels"),                            # e.g. [8,8,8] -> 512 codes
        ssl_contrastive_weight=float(cfg["loss"].get("ssl_contrastive_weight", 0.0)),
        ssl_temperature=float(cfg["loss"].get("ssl_temperature", 0.1)),
        proto_weight=float(cfg["loss"].get("proto_weight", 0.0)),             # episodic ProtoNet (loss-only)
        proto_k_shot=int(cfg["loss"].get("proto_k_shot", 4)),
        proto_temperature=float(cfg["loss"].get("proto_temperature", 1.0)),
        encoder_type=str(cfg["encoder"].get("type", "conv")),                 # "conv" or "mamba" (SSM)
    ).to(device)
    # FSQ derives its own codebook size (prod(levels)); use the model's value for all
    num_codes = int(model.num_codes)

    optimiser = optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    epochs = int(args.epochs or cfg["train"]["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg["optim"]["warmup_epochs"]) * steps_per_epoch
    base_lr = float(cfg["optim"]["lr"])
    min_lr = float(cfg["optim"]["cosine_min_lr"])
    grad_clip = float(cfg["train"]["grad_clip"])

    grace = GracefulCheckpoint()
    logger = JSONLLogger(ckpt_dir / "train.jsonl")
    init_wandb(project="har_icl", name=f"{cfg['name']}_s{args.seed}", config=cfg, offline=True)

    best_val = math.inf
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    # Early stopping: VQ-VAE val loss bottoms out very early then overfits.
    patience = int(cfg["train"].get("early_stop_patience", 0))
    epochs_no_improve = 0

    # TRANSFER-AWARE SELECTION (metric-space overfitting fix): hold N seen classes.
    tv_names = list(cfg["train"].get("transfer_val_classes", []) or [])
    selection = str(cfg["train"].get("selection", "recon"))
    tv_ids: list[int] = []
    if tv_names:
        missing = [n for n in tv_names if n not in data["label_to_int"]]
        if missing:
            raise ValueError(f"transfer_val_classes not in label_to_int: {missing}")
        tv_ids = [int(data["label_to_int"][n]) for n in tv_names]
        print(f"[transfer-val] holding out {tv_names} (ids {tv_ids}) from supervised losses; "
              f"selection={selection}")
    if selection == "transfer" and not tv_ids:
        raise ValueError("train.selection=transfer requires train.transfer_val_classes")
    tv_ids_t = torch.tensor(tv_ids, dtype=torch.long, device=device) if tv_ids else None
    X_val_np = data["X"][val_idx]
    y_val_int = np.array([int(data["label_to_int"][l]) for l in data["y"][val_idx]], dtype=np.int64)
    best_transfer = -math.inf

    def _save(path: Path, *, epoch: int, val_loss: float) -> None:
        save_torch_checkpoint(
            {
                "model_state": model.state_dict(),
                "optim_state": optimiser.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
                "label_set": label_set,
                "label_to_int": data["label_to_int"],
                "label_descriptions": data["label_descriptions"],
                "num_codes": num_codes,
                "seed": args.seed,
            },
            path,
        )

    step = 0
    history = []
    for epoch in range(epochs):
        model.train()
        ep_recon, ep_commit, ep_aux, ep_perp, n = 0.0, 0.0, 0.0, 0.0, 0
        all_indices: list[np.ndarray] = []
        for batch in train_loader:
            x, x2, label, text_emb = batch
            x = x.to(device)
            x2 = x2.to(device) if model.use_ssl else None
            label = label.to(device)
            text_emb = text_emb.to(device) if model.use_conditioning else None
            lr = _cosine_lr(step, total_steps, base_lr, min_lr, warmup_steps)
            for pg in optimiser.param_groups:
                pg["lr"] = lr
            lm = ~torch.isin(label, tv_ids_t) if tv_ids_t is not None else None
            out = model(x, text_emb=text_emb, label=label, x2=x2, loss_mask=lm)
            optimiser.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            ep_recon += float(out["recon_loss"]) * x.size(0)
            ep_commit += float(out["commit_loss"]) * x.size(0)
            ep_aux += float(out["aux_loss"]) * x.size(0)
            ep_perp += float(out["perplexity"]) * x.size(0)
            n += x.size(0)
            all_indices.append(out["indices"].detach().cpu().numpy().reshape(-1))
            step += 1
            grace.check(lambda: _save(last_path, epoch=epoch, val_loss=float("nan")))

        train_loss = ep_recon / max(1, n) + ep_commit / max(1, n)
        train_perp = ep_perp / max(1, n)
        usage = codebook_usage(np.concatenate(all_indices) if all_indices else np.array([]),
                               num_codes=num_codes)

        # Validation.
        model.eval()
        v_recon, v_commit, v_total, vn = 0.0, 0.0, 0.0, 0
        v_indices: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                x, _x2, label, text_emb = batch
                x = x.to(device)
                label = label.to(device)
                text_emb = text_emb.to(device) if model.use_conditioning else None
                # Same loss_mask as training: otherwise val_total would include aux/SupCon
                vlm = ~torch.isin(label, tv_ids_t) if tv_ids_t is not None else None
                out = model(x, text_emb=text_emb, label=label, loss_mask=vlm)   # val: recon metric only, no SSL
                v_recon += float(out["recon_loss"]) * x.size(0)
                v_commit += float(out["commit_loss"]) * x.size(0)
                v_total += float(out["loss"]) * x.size(0)
                vn += x.size(0)
                v_indices.append(out["indices"].cpu().numpy().reshape(-1))
        val_loss = (v_recon + v_commit) / max(1, vn)   # tokenizer recon quality
        val_total = v_total / max(1, vn)               # full objective incl. aux + contrastive
        transfer_f1 = None
        if tv_ids:
            transfer_f1 = _transfer_val_f1(model, X_val_np, y_val_int, tv_ids, device)
        v_usage = codebook_usage(
            np.concatenate(v_indices) if v_indices else np.array([]),
            num_codes=num_codes,
        )

        log = {
            "epoch": epoch,
            "step": step,
            "lr": lr if 'lr' in locals() else base_lr,
            "train_loss": train_loss,
            "train_perplexity": train_perp,
            "train_codebook_usage": usage,
            "val_loss": val_loss,
            "val_total_loss": val_total,
            "val_codebook_usage": v_usage,
        }
        if transfer_f1 is not None:
            log["val_transfer_f1"] = round(transfer_f1, 4)
        logger.log(**log)
        history.append(log)
        print(
            f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} "
            f"perp={train_perp:.2f}/{num_codes} "
            f"val_recon={val_loss:.4f} val_total={val_total:.4f}"
            + (f" val_transfer_f1={transfer_f1:.4f}" if transfer_f1 is not None else "")
        )

        if usage["usage_frac"] < float(cfg["train"]["early_stop_perplexity_frac"]):
            print(
                f"[warn] codebook usage {usage['usage_frac']:.2%} < threshold — possible collapse"
            )

        _save(last_path, epoch=epoch, val_loss=val_loss)
        if selection == "transfer":
            # Select/early-stop on the transfer metric — the quantity we actually want
            if transfer_f1 is not None and transfer_f1 > best_transfer:
                best_transfer = transfer_f1
                best_val = val_loss  # logged alongside for reference
                epochs_no_improve = 0
                _save(best_path, epoch=epoch, val_loss=val_loss)
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    print(f"[early-stop] no transfer-F1 improvement for {patience} epochs "
                          f"(best_transfer={best_transfer:.4f} @ ep{epoch - epochs_no_improve + 1}); "
                          f"stopping at ep{epoch+1}.")
                    break
        else:
            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                _save(best_path, epoch=epoch, val_loss=val_loss)
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    print(f"[early-stop] no val improvement for {patience} epochs "
                          f"(best_val={best_val:.4f} @ ep{epoch - epochs_no_improve + 1}); stopping at ep{epoch+1}.")
                    break

    # val_loss = reconstruction quality; val_transfer_f1 present only under transfer.
    overfit_report(history, val_loss_key="val_loss", monitor_key="val_transfer_f1")
    logger.close()
    run_id = f"{cfg['name']}_s{args.seed}"
    write_manifest(
        out_dir=Path("manifests"),
        run_id=run_id,
        config=cfg,
        ckpt_path=best_path,
        metrics={"best_val_loss": float(best_val)},
        extra={"phase": "vqvae", "seed": args.seed, "epochs": epochs},
    )
    return {"best_val_loss": float(best_val), "ckpt": str(best_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/processed/v1")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=0, help="0 = use config value")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = use config value; set lower (e.g. 128) to fit a shared GPU.")
    parser.add_argument("--split", type=str, default="C", help="Use the train/val partition of this split.")
    args = parser.parse_args()
    out = train(args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
