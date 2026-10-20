"""Phase 0 orchestrator — fuse 4 datasets into (N, 128, 6), harmonise labels"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .datasets import LOADERS
from .download import download_all
from .harmonize import (
    apply_zscore,
    fit_zscore,
    harmonise_labels,
    load_label_map,
    unified_label_set,
)
from .splits import write_splits_from_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/processed/v1")
    parser.add_argument("--download", action="store_true", help="Attempt to fetch missing data first (skips if already present).")
    parser.add_argument("--force-download", action="store_true", help="Re-download datasets even if already present.")
    parser.add_argument("--skip-missing", action="store_true", help="Continue if a dataset is unavailable.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    label_map_path = cfg["label_map"]
    if not Path(label_map_path).is_absolute():
        label_map_path = Path(args.config).parent / Path(label_map_path).name
    label_map = load_label_map(label_map_path)
    label_set = unified_label_set(label_map)
    label_to_int = {lbl: i for i, lbl in enumerate(label_set)}

    raw_dir = Path(args.raw_dir)
    if args.download:
        # Idempotent: skips datasets already on disk.
        download_all(raw_dir, force=args.force_download)

    train_pool = cfg["datasets"]["train_pool"]
    held_out = cfg["datasets"].get("held_out_dataset")
    all_datasets = list(train_pool) + ([held_out] if held_out else [])

    pieces: list[dict] = []
    for ds in all_datasets:
        if ds not in LOADERS:
            print(f"[orchestrator] unknown dataset {ds}, skipping")
            continue
        ds_path = raw_dir / ds
        if not ds_path.exists():
            msg = f"[orchestrator] {ds}: raw path {ds_path} not found"
            if args.skip_missing:
                print(msg + " — skipping")
                continue
            else:
                raise FileNotFoundError(msg + " (rerun with --download or --skip-missing)")
        try:
            X, raw_y, subj = LOADERS[ds](ds_path)
        except Exception as e:
            msg = f"[orchestrator] {ds} failed: {e}"
            if args.skip_missing:
                print(msg + " — skipping")
                continue
            raise
        y = harmonise_labels(raw_y, ds, label_map)
        keep = np.array([v is not None for v in y])
        X, y, subj = X[keep], y[keep], subj[keep]
        ds_arr = np.full(len(X), ds, dtype=object)
        pieces.append({"X": X, "y": y, "subjects": subj, "datasets": ds_arr})
        print(f"[orchestrator] {ds}: {len(X)} windows, {sorted(set(y))[:8]}{'...' if len(set(y))>8 else ''}")

    if not pieces:
        raise RuntimeError("No datasets loaded — check --raw-dir and run with --download.")

    X = np.concatenate([p["X"] for p in pieces], axis=0)
    y = np.concatenate([p["y"] for p in pieces], axis=0)
    subjects = np.concatenate([p["subjects"] for p in pieces], axis=0)
    datasets = np.concatenate([p["datasets"] for p in pieces], axis=0)

    # Subject IDs collide across datasets → namespace as "{dataset}:{sid}".
    subj_ns = np.array([f"{ds}:{s}" for ds, s in zip(datasets, subjects)], dtype=object)

    # PER-DATASET per-channel z-score.
    from .splits import make_subject_split

    subj_map = make_subject_split(subj_ns, seed=cfg["splits"]["seed"])
    is_train_subj = np.array([subj_map.get(s, "test") == "train" for s in subj_ns])
    norm_stats: dict[str, dict] = {}
    for ds in sorted(set(datasets.tolist())):
        ds_mask = datasets == ds
        # Always fit on this dataset's TRAIN subjects; only fall back to all its windows.
        fit_mask = ds_mask & is_train_subj
        if not fit_mask.any():
            fit_mask = ds_mask
        mean_ds, std_ds = fit_zscore(X, mask=fit_mask)
        X[ds_mask] = apply_zscore(X[ds_mask], mean_ds, std_ds)
        norm_stats[ds] = {"mean": mean_ds.tolist(), "std": std_ds.tolist()}
        print(f"[normalise] {ds}: per-channel std after norm "
              f"{X[ds_mask].reshape(-1, X.shape[-1]).std(0).round(3).tolist()}")
    # Representative stats for back-compat (first train-pool dataset).
    mean = np.array(norm_stats[train_pool[0]]["mean"], dtype=np.float32)
    std = np.array(norm_stats[train_pool[0]]["std"], dtype=np.float32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "windows.npz",
        X=X.astype(np.float32),
        y=np.array(y),
        subjects=np.array(subj_ns),
        datasets=np.array(datasets),
    )
    meta = {
        "label_set": label_set,
        "label_to_int": label_to_int,
        "label_descriptions": {
            lbl: label_map[lbl].get("description", lbl)
            for lbl in label_set
        },
        "channels": cfg["channels"],
        "window": cfg["window"],
        "normalisation": {
            "scheme": "per_dataset_per_channel_zscore",
            "fit_on": "per_dataset_train_subjects",
            "per_dataset": norm_stats,
            "per_channel_mean": mean.tolist(),
            "per_channel_std": std.tolist(),
        },
        "train_pool": train_pool,
        "held_out_dataset": held_out,
        "n_windows": int(len(X)),
        "datasets_per_window_unique": sorted(set(datasets.tolist())),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    write_splits_from_arrays(out_dir, labels=y, subjects=subj_ns, seed=cfg["splits"]["seed"])
    print(json.dumps({"out": str(out_dir), "n_windows": int(len(X)), "label_set": label_set}, indent=2))


if __name__ == "__main__":
    main()
