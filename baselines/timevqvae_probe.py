#!/usr/bin/env python3
"""TimeVQVAE (Lee et al., AISTATS 2023) as a HAR tokenizer — does its STFT + LF/HF"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# TimeVQVAE official package (pip install timevqvae)
from timevqvae.vqvae import VQVAE
from timevqvae.utils.nn import quantize


# macro-F1 inlined (so this runs in an isolated env, no har_icl import)
def macro_f1(golds, preds):
    if not golds:
        return 0.0
    labels = set(golds) | set(preds)
    f1s = []
    for l in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == l and p == l)
        fp = sum(1 for g, p in zip(golds, preds) if g != l and p == l)
        fn = sum(1 for g, p in zip(golds, preds) if g == l and p != l)
        f1s.append(2 * tp / max(1e-9, 2 * tp + fp + fn))
    return float(np.mean(f1s))


# data
def load_split(data_dir: Path, split: str):
    """Return per-subset dicts {X (n,L,C), labels, datasets} for train/test/support_pool."""
    d = np.load(data_dir / "windows.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)                      # (N, L, C)
    y = d["y"].astype(object)
    # dataset-of-origin per window (for the LEAK CONTROL); fall back to "?" if absent
    ds = d["datasets"].astype(object) if "datasets" in d.files else np.array(["?"] * len(X), dtype=object)
    splits = json.loads((data_dir / "splits.json").read_text())
    per_window = np.array(splits["splits"][split]["per_window"], dtype=object)
    out = {}
    for subset in ("train", "test", "support_pool"):
        idx = np.where(per_window == subset)[0]
        out[subset] = {"X": X[idx], "labels": y[idx], "datasets": ds[idx]}
    return out


def _onehot_by_value(values_by_subset):
    """Build one-hot feature vectors keyed by a categorical string (e.g."""
    vocab = sorted({str(v) for vs in values_by_subset.values() for v in vs})
    index = {v: i for i, v in enumerate(vocab)}
    feats = {}
    for sub, vs in values_by_subset.items():
        M = np.zeros((len(vs), len(vocab)), dtype=np.float32)
        for i, v in enumerate(vs):
            M[i, index[str(v)]] = 1.0
        feats[sub] = M
    return feats, vocab


# model
def build_vqvae(in_channels: int, input_length: int, n_fft: int, codebook_size: int,
                device) -> VQVAE:
    # codebook_size applied to BOTH LF and HF branches; codebook size barely moved
    m = VQVAE(
        in_channels=in_channels,
        input_length=input_length,
        n_fft=n_fft,
        init_dim=4,
        hid_dim=128,
        downsampled_width_l=8,     # LF: heavy downsample (global shape)
        downsampled_width_h=32,    # HF: light downsample (fine detail)
        encoder_n_resnet_blocks=2,
        decoder_n_resnet_blocks=2,
        codebook_size_l=codebook_size,
        codebook_size_h=codebook_size,
        kmeans_init=True,
        codebook_dim=8,
    ).to(device)
    return m


def _pool(z: torch.Tensor) -> torch.Tensor:
    """(B, D, ...) -> (B, D): mean over every spatial dim, keep batch + feature dim."""
    return z.flatten(2).mean(dim=-1) if z.dim() > 2 else z


@torch.no_grad()
def encode_features(vqvae: VQVAE, X: np.ndarray, device, batch_size: int = 256):
    """Run TimeVQVAE encoders+quantizers, return (zq_feat, zc_feat) per window."""
    vqvae.eval()
    zq_all, zc_all = [], []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[s:s + batch_size].astype(np.float32)).transpose(1, 2).contiguous().to(device)
        z_l = vqvae.encoder_l(xb)
        z_q_l, _s_l, _vl, _pl = quantize(z_l, vqvae.vq_model_l)
        z_h = vqvae.encoder_h(xb)
        z_q_h, _s_h, _vh, _ph = quantize(z_h, vqvae.vq_model_h)
        zq = torch.cat([_pool(z_q_l), _pool(z_q_h)], dim=1)
        zc = torch.cat([_pool(z_l), _pool(z_h)], dim=1)
        zq_all.append(zq.cpu().numpy())
        zc_all.append(zc.cpu().numpy())
    return np.concatenate(zq_all, 0), np.concatenate(zc_all, 0)


# unseen probe (matches codebook_recovery)
def unseen_prototype_f1(feat_by_subset, labels_by_subset, seen, unseen,
                        k_shot, max_distractors, base_seed, reps):
    """Euclidean nearest-prototype K-shot, gold + <=max_distractors seen classes"""
    import random as _random
    Ftr, ytr = feat_by_subset["train"], labels_by_subset["train"]
    Fsp, ysp = feat_by_subset["support_pool"], labels_by_subset["support_pool"]
    Fte, yte = feat_by_subset["test"], labels_by_subset["test"]

    def bin_by_label(labels):
        d = {}
        for i, l in enumerate(labels):
            d.setdefault(str(l), []).append(i)
        return d
    tr_idx, sp_idx = bin_by_label(ytr), bin_by_label(ysp)
    seen = list(seen)
    scores = []
    for r in range(max(1, reps)):
        rng = _random.Random(base_seed + 1000 * r)
        golds, preds = [], []
        for qi, gold in enumerate(yte):
            gold = str(gold)
            if gold not in set(unseen):
                continue
            others = [l for l in seen if l != gold]
            rng.shuffle(others)
            cands = others[:max_distractors] + [gold]
            rng.shuffle(cands)
            protos, plabels = [], []
            for c in cands:
                if c in tr_idx and tr_idx[c]:
                    src, F = tr_idx[c], Ftr
                elif c in sp_idx and sp_idx[c]:
                    src, F = sp_idx[c], Fsp
                else:
                    continue
                picks = [rng.choice(src) for _ in range(k_shot)]
                protos.append(F[picks].mean(axis=0))
                plabels.append(c)
            if not protos:
                continue
            P = np.stack(protos, 0)
            d = np.linalg.norm(P - Fte[qi][None, :], axis=1)
            preds.append(plabels[int(d.argmin())])
            golds.append(gold)
        scores.append(float(macro_f1(golds, preds)))
    return round(float(np.mean(scores)), 4), round(float(np.std(scores)), 4), len(scores)


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/v1")
    ap.add_argument("--split", default="C")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=8)
    ap.add_argument("--codebook_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k_shot", type=int, default=4)
    ap.add_argument("--max_distractors", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] no CUDA — TimeVQVAE training on CPU will be slow.")

    data = load_split(Path(args.data), args.split)
    L, C = data["train"]["X"].shape[1], data["train"]["X"].shape[2]
    seen = sorted(set(map(str, data["train"]["labels"].tolist())))
    unseen = sorted(set(map(str, data["support_pool"]["labels"].tolist())) - set(seen))
    print(f"[timevqvae] L={L} C={C} n_fft={args.n_fft} codebook={args.codebook_size}")
    print(f"[timevqvae] seen={len(seen)} unseen={unseen}")

    vqvae = build_vqvae(C, L, args.n_fft, args.codebook_size, device)
    opt = torch.optim.AdamW(vqvae.parameters(), lr=args.lr, weight_decay=1e-5)

    # Stage-1 training on SEEN (train) windows only
    Xtr = data["train"]["X"]
    n = len(Xtr)
    for ep in range(args.epochs):
        vqvae.train()
        perm = np.random.permutation(n)
        tot = 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb = torch.from_numpy(Xtr[idx].astype(np.float32)).transpose(1, 2).contiguous().to(device)
            out = vqvae(xb)
            recon = sum(out.recons_loss.values())
            vq = 0.0
            for v in out.vq_losses.values():
                if torch.is_tensor(v):
                    vq = vq + v.mean()
            loss = recon + vq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep+1}/{args.epochs}] train_loss={tot / max(1,n):.4f}")

    # encode all subsets -> features
    feat_zq, feat_zc, labels, dsets = {}, {}, {}, {}
    for subset in ("train", "test", "support_pool"):
        zq, zc = encode_features(vqvae, data[subset]["X"], device)
        feat_zq[subset], feat_zc[subset] = zq, zc
        labels[subset] = np.array(list(map(str, data[subset]["labels"].tolist())), dtype=object)
        dsets[subset] = np.array(list(map(str, data[subset]["datasets"].tolist())), dtype=object)

    def class_dataset_mix(target_labels):
        mix = {}
        for sub in ("train", "support_pool", "test"):
            for lab, ds in zip(labels[sub], dsets[sub]):
                mix.setdefault(lab, {}).setdefault(ds, 0)
                mix[lab][ds] += 1
        return {c: mix.get(c, {}) for c in target_labels}
    unseen_mix = class_dataset_mix(unseen)
    print("\n[leak-audit] dataset-of-origin per UNSEEN class:")
    for c, m in unseen_mix.items():
        print(f"   {c:16s} {m}")

    # unseen prototype-F1 (the comparable number)
    zq_m, zq_s, nrep = unseen_prototype_f1(
        feat_zq, labels, seen, unseen, args.k_shot, args.max_distractors, args.seed, args.reps)
    zc_m, zc_s, _ = unseen_prototype_f1(
        feat_zc, labels, seen, unseen, args.k_shot, args.max_distractors, args.seed, args.reps)

    # DATASET-ONLY CONTROL: identical protocol, features = one-hot(dataset).
    feat_ds, ds_vocab = _onehot_by_value(dsets)
    ds_m, ds_s, _ = unseen_prototype_f1(
        feat_ds, labels, seen, unseen, args.k_shot, args.max_distractors, args.seed, args.reps)

    chance = round(1.0 / (args.max_distractors + 1), 3)
    report = {
        "model": "TimeVQVAE (official)",
        "split": args.split, "n_fft": args.n_fft, "codebook_size": args.codebook_size,
        "epochs": args.epochs, "episode_reps": nrep, "chance": chance,
        "unseen_prototype_f1": {
            "zq (quantized)": [zq_m, zq_s],
            "zc (continuous)": [zc_m, zc_s],
            "CONTROL dataset-only": [ds_m, ds_s],
        },
        "dataset_vocab": ds_vocab,
        "unseen_dataset_mix": {c: m for c, m in unseen_mix.items()},
        "reference": {"vqvae_aux_contrast_t32 zq": "0.292 +/- 0.015"},
    }
    print("\n" + json.dumps(report, indent=2))
    print("\nVERDICT:")
    print(f"  TimeVQVAE zq={zq_m:.3f}  zc={zc_m:.3f}   vs winner 0.292   (chance {chance})")
    print(f"  DATASET-ONLY control = {ds_m:.3f} +/- {ds_s:.3f}   <-- the leak test")
    if ds_m > 0.30:
        print("  => LEAK LIKELY: dataset identity alone already scores high — unseen classes are")
        print("     dataset-confounded, so the TimeVQVAE number is (partly) fingerprinting, not skill.")
    elif zq_m - zq_s > 0.292 + 0.015:
        print("  => REAL GAIN: dataset-only is ~chance, so TimeVQVAE genuinely beats the ceiling.")
    else:
        print("  => dataset-only low AND no gain over winner — tie/clean, no leak.")
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)   # fix: create dir before writing
        out_p.write_text(json.dumps(report, indent=2))
        print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
