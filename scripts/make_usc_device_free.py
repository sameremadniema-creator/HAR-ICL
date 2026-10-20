#!/usr/bin/env python3
"""Build a DEVICE-FREE single-dataset processed dir for the confound-controlled experiment."""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

from har_icl.phase0_data.splits import (
    make_subject_split, make_unseen_split, assert_no_unseen_leakage, _count, SPLIT_FILE,
)
from har_icl.phase0_data.harmonize import fit_zscore, apply_zscore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="source processed dir (real v1)")
    ap.add_argument("--out", required=True, help="output device-free processed dir")
    ap.add_argument("--dataset", default="usc_had", help="source dataset tag to isolate")
    ap.add_argument("--unseen", default="elevator_up,elevator_down,walk_upstairs,walk_downstairs",
                    help="comma-separated unseen classes held out WITHIN the dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.data); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    d = np.load(src / "windows.npz", allow_pickle=True)
    X, y = d["X"], d["y"].astype(object)
    subjects = d["subjects"]; datasets = d["datasets"].astype(object)

    mask = np.asarray([str(x) == args.dataset for x in datasets])
    if mask.sum() == 0:
        raise SystemExit(f"no windows with dataset=={args.dataset!r}; "
                         f"present: {sorted(set(map(str, datasets)))}")
    Xu, yu, subju, dsu = X[mask], y[mask], subjects[mask], datasets[mask]
    present = sorted(set(map(str, yu)))
    unseen = [u for u in args.unseen.split(",") if u in present]
    missing = [u for u in args.unseen.split(",") if u not in present]
    if missing:
        print(f"[warn] requested unseen not present in {args.dataset}: {missing}")
    if not unseen:
        raise SystemExit(f"none of the requested unseen classes are in {args.dataset} "
                         f"(present: {present})")

    # subject split (70/15/15) over THIS dataset's subjects, then held-out-by-name.
    subj_map = make_subject_split(subju, seed=args.seed)
    assignment = make_unseen_split(yu, subj_map, subju, unseen)

    record = {
        "seed": int(args.seed),
        "subject_assignment": subj_map,
        "split_defs": {"C": unseen},
        "device_free": {"dataset": args.dataset, "unseen": unseen,
                        "seen": sorted(set(present) - set(unseen))},
        "splits": {"C": {"unseen_labels": unseen,
                         "per_window": assignment.tolist(),
                         "counts": _count(assignment)}},
    }
    (out / SPLIT_FILE).write_text(json.dumps(record, indent=2))
    assert_no_unseen_leakage(record, yu)   # CI gate: no unseen class in train/val

    train_subj_mask = np.array([subj_map.get(str(s), "test") == "train" for s in subju])
    norm_meta = {"refit_on": "none"}
    if train_subj_mask.any():
        mean_tr, std_tr = fit_zscore(Xu, mask=train_subj_mask)
        Xu = apply_zscore(Xu, mean_tr, std_tr)
        norm_meta = {"refit_on": "device_free_train_subjects",
                     "n_train_windows": int(train_subj_mask.sum()),
                     "mean": mean_tr.tolist(), "std": std_tr.tolist()}
        print(f"[normalise] refit on {int(train_subj_mask.sum())} train-subject windows; "
              f"per-channel std after: {Xu.reshape(-1, Xu.shape[-1]).std(0).round(3).tolist()}")

    np.savez(out / "windows.npz", X=Xu.astype(np.float32), y=np.array(yu),
             subjects=subju, datasets=np.array(dsu))

    # meta.json: carry the source meta but mark the device-free subset (source stays.
    meta = {}
    src_meta = src / "meta.json"
    if src_meta.exists():
        try: meta = json.loads(src_meta.read_text())
        except Exception: meta = {}
    meta.update({
        "device_free_subset": args.dataset,
        "n_windows": int(mask.sum()),
        "unseen_labels": unseen,
        "seen_labels": sorted(set(present) - set(unseen)),
        "note": "single-device subset for confound-controlled unseen-class HAR",
        "znorm_refit": norm_meta,
    })
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[done] {args.dataset}: {int(mask.sum())} windows -> {out}")
    print(f"  seen  ({len(record['device_free']['seen'])}): {record['device_free']['seen']}")
    print(f"  unseen({len(unseen)}): {unseen}")
    print(f"  split-C counts: {record['splits']['C']['counts']}")


if __name__ == "__main__":
    main()
