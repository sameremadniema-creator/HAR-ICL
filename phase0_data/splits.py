"""Subject-wise + held-out-by-name splits with leakage guard """
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


SPLIT_FILE = "splits.json"
STATS_FILE = "stats.json"


def _stable_subject_partition(
    unique_subjects: list,
    seed: int,
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[list, list, list]:
    """Reproducible 70/15/15 partition by subject ID."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_subjects))
    ordered = [unique_subjects[i] for i in perm]
    n = len(ordered)
    n_train = max(1, int(round(fracs[0] * n)))
    n_val = max(1, int(round(fracs[1] * n))) if n > 2 else 0
    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    if not test:
        # Tiny corpora: borrow 1 subject from train into test.
        test = train[-1:]
        train = train[:-1]
    return train, val, test


def make_subject_split(
    subjects: Sequence,
    seed: int = 42,
    fracs: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict:
    """Returns {subject -> 'train'|'val'|'test'}."""
    unique = sorted({str(s) for s in subjects})
    train, val, test = _stable_subject_partition(unique, seed, fracs)
    mapping: dict[str, str] = {}
    for s in train:
        mapping[s] = "train"
    for s in val:
        mapping[s] = "val"
    for s in test:
        mapping[s] = "test"
    return mapping


def make_unseen_split(
    labels: Sequence[str],
    subject_assignment: dict[str, str],
    subjects: Sequence,
    unseen_labels: Sequence[str],
) -> np.ndarray:
    """Build per-window split assignment for a single seen/unseen split."""
    labels = np.asarray(labels, dtype=object)
    subjects = np.asarray([str(s) for s in subjects])
    unseen = set(unseen_labels)
    out = np.empty(len(labels), dtype=object)
    for i in range(len(labels)):
        lbl = labels[i]
        subj_role = subject_assignment.get(subjects[i], "test")
        if lbl is None:
            out[i] = "excluded"
            continue
        if lbl in unseen:
            out[i] = "test" if subj_role == "test" else "support_pool"
        else:
            out[i] = subj_role
    return out


def write_splits_from_arrays(
    out_dir: str | Path,
    labels: Sequence,
    subjects: Sequence,
    seed: int = 42,
) -> dict:
    """Apply splits A, B, C to a labelled corpus and persist as splits.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subj_map = make_subject_split(subjects, seed=seed)

    # Read split definitions from configs/splits/{A,B,C}.yaml.
    from importlib.resources import files as _files

    splits_dir = Path(__file__).resolve().parent.parent / "configs" / "splits"
    split_defs: dict[str, list[str]] = {}
    for name in ("A", "B", "C"):
        p = splits_dir / f"{name}.yaml"
        cfg = yaml.safe_load(p.read_text())
        split_defs[name] = list(cfg["unseen_labels"])

    record: dict = {
        "seed": int(seed),
        "subject_assignment": subj_map,
        "split_defs": split_defs,
        "splits": {},
    }
    for name, unseen in split_defs.items():
        assignment = make_unseen_split(labels, subj_map, subjects, unseen)
        record["splits"][name] = {
            "unseen_labels": unseen,
            "per_window": assignment.tolist(),
            "counts": _count(assignment),
        }
    (out_dir / SPLIT_FILE).write_text(json.dumps(record, indent=2))
    # SHA-256 hash for reproducibility.
    h = hashlib.sha256((out_dir / SPLIT_FILE).read_bytes()).hexdigest()
    (out_dir / "sha256.txt").write_text(h + f"  {SPLIT_FILE}\n")
    return record


def _count(a: np.ndarray) -> dict:
    vals, counts = np.unique(a, return_counts=True)
    return {str(v): int(c) for v, c in zip(vals, counts)}


def load_splits(data_dir: str | Path) -> dict:
    return json.loads(Path(data_dir, SPLIT_FILE).read_text())


def assert_no_unseen_leakage(splits: dict, labels: Sequence) -> None:
    """CI gate (PDF risk table): no unseen-class window in train/val."""
    labels = np.asarray(labels, dtype=object)
    for split_name, info in splits["splits"].items():
        unseen = set(info["unseen_labels"])
        per_window = np.asarray(info["per_window"], dtype=object)
        if len(per_window) != len(labels):
            raise AssertionError(
                f"split {split_name}: per_window length {len(per_window)} != labels {len(labels)}"
            )
        train_mask = per_window == "train"
        val_mask = per_window == "val"
        bad_train = [labels[i] for i in np.where(train_mask)[0] if labels[i] in unseen]
        bad_val = [labels[i] for i in np.where(val_mask)[0] if labels[i] in unseen]
        if bad_train:
            raise AssertionError(
                f"Split {split_name}: {len(bad_train)} unseen-class windows leaked into TRAIN: "
                f"{sorted(set(bad_train))}"
            )
        if bad_val:
            raise AssertionError(
                f"Split {split_name}: {len(bad_val)} unseen-class windows leaked into VAL: "
                f"{sorted(set(bad_val))}"
            )
