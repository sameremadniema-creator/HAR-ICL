#!/usr/bin/env python3
"""preflight.py — CPU smoke test for VQ-VAE configs BEFORE submitting GPU jobs."""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import yaml
import torch

from har_icl.phase1_vqvae.model import ActivityAwareVQVAE


def resolve(p: Path) -> dict:
    c = yaml.safe_load(p.read_text())
    base = c.get("inherit")
    if base:
        b = resolve(p.parent / base)
        def merge(o, n):
            for k, v in n.items():
                o[k] = merge(o.get(k, {}), v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
            return o
        c = merge(b, c); c.pop("inherit", None)
    return c


def build(cfg: dict, n_classes: int = 17):
    e, cb, lo = cfg["encoder"], cfg["codebook"], cfg["loss"]
    return ActivityAwareVQVAE(
        in_channels=int(e["in_channels"]), hidden_channels=list(e["hidden_channels"]),
        latent_dim=int(e["latent_dim"]), num_codes=int(cb["num_codes"]),
        codebook_decay=float(cb["decay"]), commit_beta=float(lo["commit_beta"]),
        use_conditioning=bool(cfg["conditioning"]["enabled"]), text_dim=int(cfg["conditioning"]["text_dim"]),
        use_aux_classifier=bool(cfg["aux_classifier"]["enabled"]), num_aux_classes=n_classes,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    here = Path(__file__).resolve().parents[1]
    cfgs = args.configs or sorted(glob.glob(str(here / "configs/vqvae/vqvae_*.yaml")))
    torch.manual_seed(0)
    ok = True
    print(f"{'config':34s} {'tokens/win':>10} {'recon':>8} {'commit':>8} {'aux':>8} {'contrast':>9} {'proto':>8}  status")
    for cp in cfgs:
        name = Path(cp).stem
        try:
            cfg = resolve(Path(cp))
            m = build(cfg); m.train()
            B = args.batch
            n_lab = 17
            if m.use_proto:
                # The episodic proto loss needs >=2 classes with >= k_shot+1 samples each —
                B = max(B, 2 * (m.proto_k_shot + 2))
                label = torch.arange(B) % 2
            else:
                label = torch.randint(0, n_lab, (B,))
            x = torch.randn(B, args.window, args.channels)
            temb = torch.randn(B, int(cfg["conditioning"]["text_dim"])) if cfg["conditioning"]["enabled"] else None
            x2 = x.transpose(1, 2).contiguous() if m.use_ssl else None   # second view (B,C,T) for SSL
            lm = torch.ones(B, dtype=torch.bool); lm[0] = False          # exercise the loss_mask path
            out = m(x, text_emb=temb, label=label, x2=x2, loss_mask=lm)
            out["loss"].backward()
            tok = tuple(out["indices"].shape)[1]
            comps = [float(out[k]) for k in ("recon_loss", "commit_loss", "aux_loss", "contrastive_loss", "proto_loss")]
            finite = bool(torch.isfinite(out["loss"])) and all(map(lambda v: v == v, comps))
            if not finite:
                status = "NON-FINITE LOSS"
            elif m.use_proto and comps[4] == 0.0:
                finite, status = False, "PROTO LOSS NOT EXERCISED"
            else:
                status = "OK"
            ok = ok and finite
            print(f"{name:34s} {tok:>10d} {comps[0]:8.3f} {comps[1]:8.3f} {comps[2]:8.3f} {comps[3]:9.3f} {comps[4]:8.3f}  {status}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{name:34s} {'--':>10} {'--':>8} {'--':>8} {'--':>8} {'--':>9}  FAIL: {type(e).__name__}: {e}")
    print("\nPREFLIGHT", "PASSED — safe to submit." if ok else "FAILED — fix before submitting.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
