#!/usr/bin/env python3
"""diagnose_recon.py — is the VQ-VAE reconstruction actually broken, or is 163 just a"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from har_icl.phase1_vqvae.model import ActivityAwareVQVAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=512)
    args = ap.parse_args()

    d = np.load(Path(args.data) / "windows.npz", allow_pickle=True)
    X = d["X"].astype(np.float32)
    print(f"X shape {X.shape}")
    flat = X.reshape(-1, X.shape[-1])
    print("per-channel mean:", np.round(flat.mean(0), 3))
    print("per-channel std :", np.round(flat.std(0), 3), " <-- should be ~1.0 if z-scored")
    print("global var:", round(float(X.var()), 3))

    ck = torch.load(args.ckpt, map_location="cpu")
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
        encoder_strides=c["encoder"].get("strides"),
        quantizer_type=str(c["codebook"].get("type", "vq")),
        fsq_levels=c["codebook"].get("levels"),
        ssl_contrastive_weight=float(c["loss"].get("ssl_contrastive_weight", 0.0)),
        ssl_temperature=float(c["loss"].get("ssl_temperature", 0.1)),
        encoder_type=str(c["encoder"].get("type", "conv")),
    )
    m.load_state_dict(ck["model_state"])
    m.eval()

    idx = np.random.default_rng(0).choice(len(X), size=min(args.n, len(X)), replace=False)
    xb = torch.tensor(X[idx]).transpose(1, 2)  # (B, 6, 128)
    with torch.no_grad():
        out = m(xb)                            # no text_emb -> recon-only sanity
    xh = out["x_hat"]
    mse = float(((xb - xh) ** 2).mean())
    var = float(xb.var())
    # Per-channel variance-explained R^2.
    num = ((xb - xh) ** 2).mean(dim=(0, 2))
    den = xb.var(dim=(0, 2)) + 1e-8
    r2 = (1.0 - num / den).numpy()
    print("\n--- reconstruction ---")
    print("raw MSE              :", round(mse, 4))
    print("data variance        :", round(var, 4))
    print("NORMALISED MSE (MSE/var):", round(mse / var, 4), " <-- <0.5 good, ~1 = no better than mean, >>1 = broken")
    print("per-channel R^2      :", np.round(r2, 3), " <-- closer to 1.0 is better")
    print("x_hat std            :", round(float(xh.std()), 3), "(should be close to x std ~1)")


if __name__ == "__main__":
    main()
