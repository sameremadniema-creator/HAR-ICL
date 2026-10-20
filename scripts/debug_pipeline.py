#!/usr/bin/env python3
"""debug_pipeline.py — CPU round-trip smoke test BEFORE submitting GPU jobs."""
from __future__ import annotations

import glob
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from har_icl.phase1_vqvae.model import ActivityAwareVQVAE
from har_icl.phase2_tokenize.encode_signals import _build_model_from_ckpt
from har_icl.phase4_eval.codebook_recovery import reconstruction, _zq_mean, _encoder_z_mean

N_CLASSES = 17
WINDOW = 128
CHANNELS = 6


def resolve(p: Path) -> dict:
    """Load a yaml config and merge its `inherit` parent (same logic as preflight)."""
    c = yaml.safe_load(p.read_text())
    base = c.get("inherit")
    if base:
        b = resolve(p.parent / base)

        def merge(o, n):
            for k, v in n.items():
                o[k] = merge(o.get(k, {}), v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
            return o

        c = merge(b, c)
        c.pop("inherit", None)
    return c


def build(cfg: dict) -> ActivityAwareVQVAE:
    e, cb, lo = cfg["encoder"], cfg["codebook"], cfg["loss"]
    return ActivityAwareVQVAE(
        in_channels=int(e["in_channels"]), hidden_channels=list(e["hidden_channels"]),
        latent_dim=int(e["latent_dim"]), num_codes=int(cb["num_codes"]),
        codebook_decay=float(cb["decay"]), commit_beta=float(lo["commit_beta"]),
        use_conditioning=bool(cfg["conditioning"]["enabled"]), text_dim=int(cfg["conditioning"]["text_dim"]),
        use_aux_classifier=bool(cfg["aux_classifier"]["enabled"]), num_aux_classes=N_CLASSES,
        recon_l1_weight=float(lo["recon_l1_weight"]), aux_ce_weight=float(lo["aux_ce_weight"]),
        kernel_size=int(e.get("kernel_size", 5)), encoder_strides=e.get("strides"),
        contrastive_weight=float(lo.get("contrastive_weight", 0.0)),
        contrastive_temperature=float(lo.get("contrastive_temperature", 0.1)),
        quantizer_type=str(cb.get("type", "vq")), fsq_levels=cb.get("levels"),
        ssl_contrastive_weight=float(lo.get("ssl_contrastive_weight", 0.0)),
        ssl_temperature=float(lo.get("ssl_temperature", 0.1)),
        proto_weight=float(lo.get("proto_weight", 0.0)),
        proto_k_shot=int(lo.get("proto_k_shot", 4)),
        proto_temperature=float(lo.get("proto_temperature", 1.0)),
        encoder_type=str(e.get("type", "conv")),
    )


def expected_tokens(cfg: dict) -> int:
    strides = cfg["encoder"].get("strides") or [2] * len(cfg["encoder"]["hidden_channels"])
    return WINDOW // int(math.prod(int(s) for s in strides))


def make_ckpt_dict(model: ActivityAwareVQVAE, cfg: dict) -> dict:
    """Mirror train.py's _save payload exactly (only the keys loaders read)."""
    label_set = [f"act{i}" for i in range(N_CLASSES)]
    return {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "epoch": 0,
        "val_loss": 0.0,
        "label_set": label_set,
        "label_to_int": {l: i for i, l in enumerate(label_set)},
        "label_descriptions": {l: l for l in label_set},
        "num_codes": int(model.num_codes),
        "seed": 0,
    }


def check_one(cfg_path: str, tmp: Path) -> tuple[bool, str]:
    cfg = resolve(Path(cfg_path))
    torch.manual_seed(0)

    # (1)+(2) build + forward/backward
    m = build(cfg)
    m.train()
    x = torch.randn(8, WINDOW, CHANNELS)
    label = torch.randint(0, N_CLASSES, (8,))
    temb = torch.randn(8, int(cfg["conditioning"]["text_dim"])) if cfg["conditioning"]["enabled"] else None
    x2 = x.transpose(1, 2).contiguous() if m.use_ssl else None   # second view (B,C,T) exercises NT-Xent
    out = m(x, text_emb=temb, label=label, x2=x2)
    out["loss"].backward()
    if not torch.isfinite(out["loss"]):
        return False, "non-finite loss"

    n_codes = int(m.num_codes)
    n_tok = expected_tokens(cfg)

    # (3) save in the real checkpoint format
    ckpt = make_ckpt_dict(m, cfg)
    ckpt_path = tmp / "best.pt"
    torch.save(ckpt, ckpt_path)

    # (4a) reload through the REAL phase-2 loader (this is what crashed for FSQ/t32)
    reloaded = _build_model_from_ckpt(ckpt, num_classes=N_CLASSES)  # calls load_state_dict internally
    reloaded.eval()

    # (5) encode + index sanity
    with torch.no_grad():
        z_q, idx = reloaded.encode(torch.randn(4, CHANNELS, WINDOW))
    if tuple(idx.shape)[1] != n_tok:
        return False, f"tokens/win={tuple(idx.shape)[1]} expected {n_tok}"
    imin, imax = int(idx.min()), int(idx.max())
    if imin < 0 or imax >= n_codes:
        return False, f"index out of range [{imin},{imax}] not in [0,{n_codes})"

    # (4b)+(6) reconstruction loader returns the codebook (VQ: stored; FSQ.
    X_dbg = np.random.randn(40, WINDOW, CHANNELS).astype(np.float32)
    rec = reconstruction(ckpt_path, X_dbg, n=20, seed=0)
    emb = rec["embedding"]
    latent = int(cfg["encoder"]["latent_dim"])
    if emb.shape != (n_codes, latent):
        return False, f"codebook shape {emb.shape} expected ({n_codes},{latent})"
    # zq_mean feature path used by the probe
    feats = _zq_mean(idx.numpy(), emb)
    if feats.shape != (4, latent):
        return False, f"zq_mean shape {feats.shape} expected (4,{latent})"
    # continuous z_mean diagnostic path (origin ids -> raw windows -> encoder ->.
    zf = _encoder_z_mean(rec["model"], X_dbg, np.arange(8))
    if zf.shape != (8, latent):
        return False, f"z_mean shape {zf.shape} expected (8,{latent})"

    return True, f"tok={n_tok} codes={n_codes} normMSE={rec['normalised_mse']:.3f}"


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    cfgs = sys.argv[1:] or sorted(glob.glob(str(here / "configs/vqvae/vqvae_*.yaml")))
    ok = True
    print(f"{'config':34s} {'status':6s}  detail")
    for cp in cfgs:
        name = Path(cp).stem
        with tempfile.TemporaryDirectory() as td:
            try:
                passed, detail = check_one(cp, Path(td))
            except Exception as e:  # noqa: BLE001
                passed, detail = False, f"{type(e).__name__}: {e}"
        ok = ok and passed
        print(f"{name:34s} {'OK' if passed else 'FAIL':6s}  {detail}")
    print("\nDEBUG PASSED — round-trip clean, safe to submit." if ok
          else "\nDEBUG FAILED — fix before submitting.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
