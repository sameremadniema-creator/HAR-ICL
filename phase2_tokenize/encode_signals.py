"""Run frozen VQ-VAE over harmonised windows → save codebook indices per split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from har_icl.phase1_vqvae.model import ActivityAwareVQVAE
from har_icl.phase1_vqvae.conditioning import fallback_random_embedding
from har_icl.utils.checkpointing import load_torch_checkpoint


SUBSETS = ("train", "val", "test", "support_pool")


def _build_model_from_ckpt(ckpt: dict, num_classes: int) -> ActivityAwareVQVAE:
    cfg = ckpt["cfg"]
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
        num_aux_classes=num_classes,
        recon_l1_weight=float(cfg["loss"]["recon_l1_weight"]),
        aux_ce_weight=float(cfg["loss"]["aux_ce_weight"]),
        kernel_size=int(cfg["encoder"].get("kernel_size", 5)),
        encoder_strides=cfg["encoder"].get("strides"),                # match t32 architecture
        quantizer_type=str(cfg["codebook"].get("type", "vq")),       # match FSQ vs VQ checkpoint
        fsq_levels=cfg["codebook"].get("levels"),
        ssl_contrastive_weight=float(cfg["loss"].get("ssl_contrastive_weight", 0.0)),  # rebuild ssl_proj if present
        ssl_temperature=float(cfg["loss"].get("ssl_temperature", 0.1)),
        encoder_type=str(cfg["encoder"].get("type", "conv")),        # match conv vs mamba checkpoint
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def encode_split(
    model: ActivityAwareVQVAE,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    datasets_arr: np.ndarray,
    per_window: np.ndarray,
    out_dir: Path,
    text_embs: dict[str, np.ndarray] | None,
    device: torch.device,
    use_conditioning: bool,
    batch_size: int = 256,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for subset in SUBSETS:
        mask = per_window == subset
        idx = np.where(mask)[0]
        if len(idx) == 0:
            counts[subset] = 0
            continue
        X_sub = X[idx]
        y_sub = y[idx]
        all_indices: list[np.ndarray] = []
        for start in range(0, len(idx), batch_size):
            chunk = X_sub[start : start + batch_size]
            chunk_y = y_sub[start : start + batch_size]
            xb = torch.from_numpy(chunk).transpose(1, 2).contiguous().to(device)
            if use_conditioning and text_embs is not None:
                te = np.stack(
                    [text_embs.get(lbl, np.zeros(next(iter(text_embs.values())).shape, dtype=np.float32))
                     for lbl in chunk_y],
                    axis=0,
                )
                te = torch.from_numpy(te.astype(np.float32)).to(device)
            else:
                te = None
            with torch.no_grad():
                _, ind = model.encode(xb, text_emb=te)
            all_indices.append(ind.cpu().numpy().astype(np.int16))
        indices = np.concatenate(all_indices, axis=0) if all_indices else np.empty((0, 16), dtype=np.int16)
        np.savez(
            out_dir / f"{subset}.npz",
            indices=indices,
            labels=y_sub,
            subjects=subjects[idx],
            datasets=datasets_arr[idx],
            origin=idx.astype(np.int64),
        )
        counts[subset] = int(len(idx))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/processed/v1")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--allow_conditioning", action="store_true",
                        help="opt in to name-conditioned tokenization. UNSAFE for held-out-class splits: "
                             "it feeds each window's TRUE activity name into the encoder, leaking the label "
                             "(including for unseen/test windows) into the codes.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_torch_checkpoint(args.vqvae_ckpt, map_location=str(device))

    data_dir = Path(args.data)
    data = np.load(data_dir / "windows.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(object)
    subjects = data["subjects"].astype(object)
    datasets_arr = data["datasets"].astype(object)
    meta = json.loads((data_dir / "meta.json").read_text())
    splits = json.loads((data_dir / "splits.json").read_text())

    label_set = ckpt.get("label_set") or meta["label_set"]
    num_classes = len(label_set)
    model = _build_model_from_ckpt(ckpt, num_classes=num_classes).to(device)

    if model.use_conditioning and not args.allow_conditioning:
        raise SystemExit(
            "[leakage guard] this checkpoint has name-conditioning ENABLED, which feeds each window's "
            "true activity name into the encoder at tokenize time — leaking the label (incl. unseen/test "
            "windows) into the codes. Re-train with conditioning off (the device-free experiment uses "
            "vqvae_aux_contrast_t32, conditioning=false), or pass --allow_conditioning if you really mean it."
        )

    text_embs = None
    if model.use_conditioning:
        # Load cached embeddings from ckpt dir, fall back to deterministic random.
        cache_path = Path(args.vqvae_ckpt).parent / "text_emb.npz"
        if cache_path.exists():
            d = np.load(cache_path, allow_pickle=True)
            keys = list(d["keys"])
            embs = d["embeddings"]
            text_embs = {k: embs[i].astype(np.float32) for i, k in enumerate(keys)}
        else:
            text_embs = fallback_random_embedding(label_set, dim=int(ckpt["cfg"]["conditioning"]["text_dim"]))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"vqvae_ckpt": str(args.vqvae_ckpt), "splits": {}}
    for split in args.splits:
        per_window = np.array(splits["splits"][split]["per_window"], dtype=object)
        out_dir = out_root / f"split_{split}"
        counts = encode_split(
            model, X, y, subjects, datasets_arr, per_window, out_dir,
            text_embs=text_embs, device=device,
            use_conditioning=bool(model.use_conditioning),
            batch_size=args.batch_size,
        )
        summary["splits"][split] = counts
        print(f"[tokenize] split={split} counts={counts}")
    # Persist label set + num_codes alongside tokens so downstream needn't peek at.
    (out_root / "meta.json").write_text(json.dumps({
        "label_set": label_set,
        "label_to_int": ckpt.get("label_to_int") or meta.get("label_to_int", {}),
        "label_descriptions": ckpt.get("label_descriptions") or meta.get("label_descriptions", {}),
        "num_codes": int(ckpt.get("num_codes") or ckpt["cfg"]["codebook"]["num_codes"]),
        "vqvae_ckpt": str(args.vqvae_ckpt),
        "splits": list(args.splits),
    }, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
