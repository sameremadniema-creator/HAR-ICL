#!/usr/bin/env python3
"""Codebook task-recovery / upper-bound analysis (signals analogue of Reiss et al."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import torch

from har_icl.utils.metrics import macro_f1


# token features
def _load_subset(tokens_dir: Path, split: str, subset: str):
    p = tokens_dir / f"split_{split}" / f"{subset}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    out = {"indices": d["indices"].astype(np.int64), "labels": d["labels"].astype(object)}
    # 'origin' = original window row ids (written by encode_signals) — lets us pull.
    if "origin" in d.files:
        out["origin"] = d["origin"].astype(np.int64)
    # 'datasets' = source-dataset tag per window — needed for the device-constant
    if "datasets" in d.files:
        out["datasets"] = d["datasets"].astype(object)
    return out


def _filter_subset(sub: dict, mask: np.ndarray) -> dict:
    """Keep only rows where mask is True, across every per-row array in the subset."""
    return {k: v[mask] for k, v in sub.items()}


def name_that_dataset(feats_by_sub: dict, subsets: dict, seed: int = 42) -> dict | None:
    """Torralba-Efros (CVPR 2011) 'name that dataset' confound control: can the DATASET-of-origin"""
    if any("datasets" not in s for s in subsets.values()):
        return None
    from sklearn.linear_model import LogisticRegression
    Xs, ys = [], []
    for name in subsets:
        Xs.append(feats_by_sub[name]["zq_mean"])
        ys.append(np.asarray([str(x) for x in subsets[name]["datasets"]]))
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)
    n_ds = len(set(y.tolist()))
    if n_ds < 2:
        return {"name_that_dataset_acc": None, "chance": None, "n_datasets": n_ds,
                "note": "single dataset present (device-constant) — control is degenerate by design"}
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    cut = int(0.7 * len(X)); tr, te = idx[:cut], idx[cut:]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
    acc = float((clf.predict(X[te]) == y[te]).mean())
    return {"name_that_dataset_acc": round(acc, 4), "chance": round(1.0 / n_ds, 4),
            "n_datasets": n_ds, "interpretation": "acc >> chance => exploitable device fingerprint"}


def _histograms(indices: np.ndarray, num_codes: int) -> np.ndarray:
    """(N, T) code indices -> (N, num_codes) L1-normalised bag-of-codes."""
    N = indices.shape[0]
    H = np.zeros((N, num_codes), dtype=np.float32)
    for i in range(N):
        bc = np.bincount(indices[i], minlength=num_codes).astype(np.float32)
        s = bc.sum()
        H[i] = bc / s if s > 0 else bc
    return H


def _zq_mean(indices: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    """(N, T) indices -> (N, D) mean codebook vector per window."""
    return embedding[indices].mean(axis=1).astype(np.float32)


def _features(sub, num_codes, embedding):
    return {"hist": _histograms(sub["indices"], num_codes),
            "zq_mean": _zq_mean(sub["indices"], embedding)}


def _fit_whiten_l2(F: np.ndarray, shrink: float = 0.1):
    """Fit shrinkage-ZCA whitening (SimpleShot-style)."""
    mu = F.mean(axis=0)
    Fc = F - mu
    cov = (Fc.T @ Fc) / max(1, len(F))
    d = cov.shape[0]
    cov = cov + shrink * (np.trace(cov) / d) * np.eye(d)   # shrinkage (robust on small pools)
    vals, vecs = np.linalg.eigh(cov)
    W = (vecs @ np.diag(1.0 / np.sqrt(vals + 1e-8)) @ vecs.T).astype(np.float32)
    return mu.astype(np.float32), W


def _apply_whiten_l2(F: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    Fw = (F - mu) @ W
    n = np.linalg.norm(Fw, axis=1, keepdims=True) + 1e-8
    return (Fw / n).astype(np.float32)


def _encoder_z_mean(model, X: np.ndarray, origin: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """CONTINUOUS pre-quantizer features: run the frozen encoder on the subset's raw"""
    feats = []
    with torch.no_grad():
        for s in range(0, len(origin), batch_size):
            chunk = X[origin[s : s + batch_size]].astype(np.float32)
            xb = torch.from_numpy(chunk).transpose(1, 2).contiguous()   # (B, C, T)
            z_e = model.encoder(xb)                                      # (B, D, T')
            feats.append(z_e.mean(dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# (B) class recovery
def seen_linear_probe(train, test, feats_train, feats_test, seen_labels) -> dict:
    from sklearn.linear_model import LogisticRegression
    out = {}
    seen = set(seen_labels)
    te_mask = np.array([l in seen for l in test["labels"]])
    for key in sorted(feats_train.keys()):
        Xtr, ytr = feats_train[key], train["labels"]
        Xte, yte = feats_test[key][te_mask], test["labels"][te_mask]
        clf = LogisticRegression(max_iter=2000, C=1.0)  # multinomial auto-handled in sklearn>=1.7
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        out[key] = round(float(macro_f1(list(yte), list(pred))), 4)
    return out


def unseen_prototype_fewshot(test, support_pool, train, feats, k_shot, max_distractors,
                             seen_labels, unseen_labels, seed) -> dict:
    """Model-free K-shot nearest-prototype in token space, episode protocol matched to ICL."""
    seen = list(seen_labels)
    # index lists per label in each source
    def bin_by_label(labels):
        d = {}
        for i, l in enumerate(labels):
            d.setdefault(str(l), []).append(i)
        return d
    tr_idx = bin_by_label(train["labels"])
    sp_idx = bin_by_label(support_pool["labels"])
    results = {}
    for key in sorted(feats["test"].keys()):
        # Re-seed PER FEATURE KEY: every feature is scored on the IDENTICAL episode set
        rng = random.Random(seed)
        Ftest = feats["test"][key]
        Ftr = feats["train"][key]
        Fsp = feats["support"][key]
        golds, preds = [], []
        for qi, gold in enumerate(test["labels"]):
            gold = str(gold)
            if gold not in set(unseen_labels):
                continue
            # candidates = gold + up to max_distractors seen classes
            others = [l for l in seen if l != gold]
            rng.shuffle(others)
            cands = others[:max_distractors] + [gold]
            rng.shuffle(cands)
            protos, plabels = [], []
            for c in cands:
                if c in tr_idx and tr_idx[c]:
                    src_idx, F = tr_idx[c], Ftr
                elif c in sp_idx and sp_idx[c]:
                    src_idx, F = sp_idx[c], Fsp
                else:
                    continue
                picks = [rng.choice(src_idx) for _ in range(k_shot)]
                protos.append(F[picks].mean(axis=0))
                plabels.append(c)
            if not protos:
                continue
            P = np.stack(protos, axis=0)
            q = Ftest[qi]
            d = np.linalg.norm(P - q[None, :], axis=1)   # euclidean to each prototype
            pred = plabels[int(d.argmin())]
            golds.append(gold); preds.append(pred)
        results[key] = round(float(macro_f1(golds, preds)), 4)
        results[key + "_n"] = len(golds)
        # Closed-set accuracy (did the episode pick the gold among its candidates?).
        results[key + "_acc"] = (round(float(np.mean([1.0 if g == p else 0.0
                                 for g, p in zip(golds, preds)])), 4) if golds else 0.0)
        # Per-class unseen F1 — separates genuinely-transferring classes (e.g.
        from sklearn.metrics import f1_score
        cls = sorted(set(golds))
        pcf = f1_score(golds, preds, labels=cls, average=None, zero_division=0)
        results[key + "_per_class"] = {c: round(float(v), 4) for c, v in zip(cls, pcf)}
        # Macro-F1 over the UNSEEN classes ONLY (field-standard for unseen-class.
        results[key + "_unseen_macro"] = round(float(np.mean(pcf)), 4) if len(pcf) else 0.0
    return results


# (A) reconstruction
def reconstruction(ckpt_path: Path, X: np.ndarray, n: int, seed: int) -> dict:
    from har_icl.phase1_vqvae.model import ActivityAwareVQVAE
    ck = torch.load(ckpt_path, map_location="cpu")
    c = ck["cfg"]
    m = ActivityAwareVQVAE(
        in_channels=int(c["encoder"]["in_channels"]),
        hidden_channels=list(c["encoder"]["hidden_channels"]),
        latent_dim=int(c["encoder"]["latent_dim"]),
        num_codes=int(c["codebook"]["num_codes"]),
        codebook_decay=float(c["codebook"]["decay"]),
        commit_beta=float(c["loss"]["commit_beta"]),
        use_conditioning=bool(c["conditioning"]["enabled"]),
        text_dim=int(c["conditioning"]["text_dim"]),
        use_aux_classifier=bool(c["aux_classifier"]["enabled"]),
        num_aux_classes=len(ck.get("label_set", [])) or 17,
        recon_l1_weight=float(c["loss"]["recon_l1_weight"]),
        aux_ce_weight=float(c["loss"]["aux_ce_weight"]),
        kernel_size=int(c["encoder"].get("kernel_size", 5)),
        encoder_strides=c["encoder"].get("strides"),                 # match t32 architecture
        quantizer_type=str(c["codebook"].get("type", "vq")),        # match FSQ vs VQ checkpoint
        fsq_levels=c["codebook"].get("levels"),
        ssl_contrastive_weight=float(c["loss"].get("ssl_contrastive_weight", 0.0)),  # rebuild ssl_proj if present
        ssl_temperature=float(c["loss"].get("ssl_temperature", 0.1)),
        encoder_type=str(c["encoder"].get("type", "conv")),          # match conv vs mamba checkpoint
    )
    m.load_state_dict(ck["model_state"]); m.eval()
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    xb = torch.tensor(X[idx]).transpose(1, 2)
    with torch.no_grad():
        out = m(xb)
    xh = out["x_hat"]
    mse = float(((xb - xh) ** 2).mean()); var = float(xb.var())
    num = ((xb - xh) ** 2).mean(dim=(0, 2)); den = xb.var(dim=(0, 2)) + 1e-8
    return {
        "normalised_mse": round(mse / var, 4),
        "mae": round(float((xb - xh).abs().mean()), 4),
        "per_channel_r2": [round(float(v), 3) for v in (1 - num / den).tolist()],
        # VQ stores its codebook directly; FSQ has none — materialise its implicit.
        "embedding": (
            ck["model_state"]["quantizer.embedding"].cpu().numpy()
            if "quantizer.embedding" in ck["model_state"]
            else m.quantizer.codebook().cpu().numpy()
        ),
        # Hand the frozen model back so main() can compute the continuous z_mean
        "model": m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="C")
    ap.add_argument("--k_shot", type=int, default=4)
    ap.add_argument("--max_distractors", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episode_reps", type=int, default=5,
                    help="repeat the unseen episode protocol over N seeds; report mean+std "
                         "(single-draw differences of ~0.02 are episode noise, not signal)")
    ap.add_argument("--recon_n", type=int, default=1000)
    ap.add_argument("--restrict_dataset", default=None,
                    help="DEVICE-CONSTANT confound control: restrict train/test/support to ONE "
                         "source dataset (e.g. pamap2, usc_had). Unseen classes are then held out "
                         "WITHIN that dataset, so the device is constant and a device-fingerprint "
                         "shortcut is impossible by construction. If unseen prototype-F1 stays well "
                         "above chance here, the recognition is genuine activity transfer, not the "
                         "dataset confound. Requires 'datasets' field in the token npz.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokens_dir = Path(args.tokens)
    meta = json.loads((tokens_dir / "meta.json").read_text())
    num_codes = int(meta.get("num_codes", 512))

    train = _load_subset(tokens_dir, args.split, "train")
    test = _load_subset(tokens_dir, args.split, "test")
    support = _load_subset(tokens_dir, args.split, "support_pool") or train
    if train is None or test is None:
        raise FileNotFoundError(f"missing token subsets under {tokens_dir}/split_{args.split}")

    # DEVICE-CONSTANT confound control: restrict every subset to one source dataset.
    if args.restrict_dataset:
        rd = str(args.restrict_dataset)
        for nm, s in (("train", train), ("test", test), ("support", support)):
            if "datasets" not in s:
                raise SystemExit(f"[restrict] '{nm}' token npz lacks a 'datasets' field — "
                                 f"re-run phase2_tokenize/encode_signals.py to regenerate tokens.")
        def _ds_mask(s):
            return np.asarray([str(x) == rd for x in s["datasets"]])
        train = _filter_subset(train, _ds_mask(train))
        test = _filter_subset(test, _ds_mask(test))
        support = _filter_subset(support, _ds_mask(support))
        print(f"[restrict] dataset={rd}: train={len(train['labels'])} "
              f"test={len(test['labels'])} support={len(support['labels'])}")
        if len(test["labels"]) == 0 or len(support["labels"]) == 0:
            raise SystemExit(f"[restrict] no windows left for dataset={rd} — check the tag spelling "
                             f"(expected one of the meta.json dataset names).")

    seen_labels = sorted(set(train["labels"].tolist()))
    unseen_labels = sorted(set(support["labels"].tolist()) - set(seen_labels))

    # Raw windows: needed by (A) reconstruction and by the continuous z_mean.
    X_all = np.load(Path(args.data) / "windows.npz", allow_pickle=True)["X"].astype(np.float32)

    # (A) reconstruction (also returns codebook embedding for zq_mean features + the.
    rec = reconstruction(Path(args.ckpt), X_all, args.recon_n, args.seed)
    embedding = rec.pop("embedding")
    model = rec.pop("model")
    num_codes = int(embedding.shape[0])

    feats = {
        "train": _features(train, num_codes, embedding),
        "test": _features(test, num_codes, embedding),
        "support": _features(support, num_codes, embedding),
    }
    # SimpleShot-style whitening (L2 + shrinkage-ZCA of zq_mean, fit on.
    _fitF = np.concatenate([feats["train"]["zq_mean"], feats["support"]["zq_mean"]], axis=0)
    _mu, _W = _fit_whiten_l2(_fitF)
    for _sub in ("train", "test", "support"):
        feats[_sub]["zq_mean_w"] = _apply_whiten_l2(feats[_sub]["zq_mean"], _mu, _W)

    # CONTINUOUS pre-quantizer diagnostic (z_mean / z_mean_w): is the quantizer or the
    subsets = {"train": train, "test": test, "support": support}
    if all("origin" in s for s in subsets.values()):
        for name, s in subsets.items():
            feats[name]["z_mean"] = _encoder_z_mean(model, X_all, s["origin"])
        _fitZ = np.concatenate([feats["train"]["z_mean"], feats["support"]["z_mean"]], axis=0)
        _muz, _Wz = _fit_whiten_l2(_fitZ)
        for _sub in ("train", "test", "support"):
            feats[_sub]["z_mean_w"] = _apply_whiten_l2(feats[_sub]["z_mean"], _muz, _Wz)
    else:
        print("[warn] token npz files lack 'origin' ids — skipping continuous z_mean "
              "diagnostic (re-run phase2_tokenize/encode_signals.py to regenerate).")

    # DATASET-ONLY baseline feature: one-hot of the source-dataset tag, scored.
    if all("datasets" in s for s in subsets.values()):
        vocab = sorted(set().union(*[{str(x) for x in s["datasets"]} for s in subsets.values()]))
        vi = {d: i for i, d in enumerate(vocab)}
        for name, s in subsets.items():
            oh = np.zeros((len(s["datasets"]), len(vocab)), dtype=np.float32)
            for j, x in enumerate(s["datasets"]):
                oh[j, vi[str(x)]] = 1.0
            feats[name]["dataset_only"] = oh

    # EMPIRICAL CHANCE baseline: random Gaussian features scored through the.
    _rngc = np.random.default_rng(args.seed)
    for name in ("train", "test", "support"):
        n = feats[name]["zq_mean"].shape[0]
        feats[name]["random"] = _rngc.standard_normal((n, 16)).astype(np.float32)

    seen_probe = seen_linear_probe(train, test, feats["train"], feats["test"], seen_labels)
    # Repeat the episode protocol over several seeds: a single draw has ~±0.02 noise
    runs = [
        unseen_prototype_fewshot(
            test, support, train, feats, args.k_shot, args.max_distractors,
            seen_labels, unseen_labels, args.seed + 1000 * r)
        for r in range(max(1, args.episode_reps))
    ]
    unseen_proto = {}
    for key in runs[0]:
        if key.endswith("_n"):
            unseen_proto[key] = runs[0][key]
            continue
        if key.endswith("_per_class"):
            # average each class's F1 across episode reps
            classes = runs[0][key].keys()
            unseen_proto[key] = {c: round(float(np.mean([r[key][c] for r in runs])), 4)
                                 for c in classes}
            continue
        vals = np.array([r[key] for r in runs], dtype=np.float64)
        unseen_proto[key] = round(float(vals.mean()), 4)
        unseen_proto[key + "_std"] = round(float(vals.std()), 4)

    # Torralba-Efros confound control (cross-dataset only): how much device.
    ntd = (name_that_dataset(feats, {"train": train, "test": test, "support": support},
                             seed=args.seed) if not args.restrict_dataset else None)

    report = {
        "split": args.split,
        "restrict_dataset": args.restrict_dataset,
        "name_that_dataset_control": ntd,
        "num_codes": num_codes,
        "seen_labels": seen_labels,
        "unseen_labels": unseen_labels,
        "episode_protocol": {
            "k_shot": args.k_shot,
            "max_distractors": args.max_distractors,
            "episode_reps": int(max(1, args.episode_reps)),
            "base_seed": args.seed,
        },
        "reconstruction": rec,
        "seen_linear_probe_macro_f1": seen_probe,
        "unseen_prototype_fewshot_macro_f1": unseen_proto,
    }
    print(json.dumps(report, indent=2))
    out = Path(args.out) if args.out else tokens_dir / f"codebook_recovery_split{args.split}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[saved] {out}")
    print("\nINTERPRETATION:")
    print("  unseen prototype-F1 >> GPT-2 unseen-F1  => codebook OK, the LLM/ICL is the bottleneck.")
    print("  unseen prototype-F1 ~  GPT-2 unseen-F1  => the CODEBOOK is the bottleneck (improve it).")
    uz, um = unseen_proto.get("zq_mean"), unseen_proto.get("z_mean")
    if uz is not None and um is not None:
        uzw = unseen_proto.get("zq_mean_w") or uz
        umw = unseen_proto.get("z_mean_w") or um
        gap = max(um, umw) - max(uz, uzw)
        print(f"\nBOTTLENECK DIAGNOSTIC: continuous z_mean={um:.3f} (w={umw:.3f}) "
              f"vs quantized zq_mean={uz:.3f} (w={uzw:.3f})  [gap {gap:+.3f}]")
        if gap > 0.03:
            print("  => the QUANTIZER is destroying unseen info: try RVQ / finer codes / "
                  "distance-preserving quantization.")
        else:
            print("  => the ENCODER is the ceiling: quantizer tweaks won't help — invest in the "
                  "representation (label-free SSL, pretrained-encoder distillation).")


if __name__ == "__main__":
    main()
