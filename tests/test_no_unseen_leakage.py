"""CI gate (PDF risk table): no unseen-class window may appear in TRAIN or VAL."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from har_icl.phase0_data.splits import (
    assert_no_unseen_leakage,
    make_subject_split,
    write_splits_from_arrays,
)

# Unified label set (kept in sync with configs/label_map_strict.yaml).
UNIFIED_LABELS = [
    "walk", "walk_upstairs", "walk_downstairs",
    "sit", "stand", "lie",
    "run", "cycle", "nordic_walk",
    "iron", "vacuum", "rope_jump",
    "elevator_up", "elevator_down",
    "arm_elevation", "knees_bend", "waist_bend",
]


def _label_subject_fixture(n_per_class: int = 32, n_subjects: int = 8, seed: int = 7):
    """Return (labels, subjects) arrays covering every unified label."""
    rng = np.random.default_rng(seed)
    labels, subjects = [], []
    for lbl in UNIFIED_LABELS:
        for _ in range(n_per_class):
            labels.append(lbl)
            subjects.append(int(rng.integers(0, n_subjects)))
    return np.array(labels, dtype=object), np.array(subjects, dtype=np.int32)


def test_subject_split_partitions_every_subject():
    subjects = [f"d:{i}" for i in range(20)]
    mapping = make_subject_split(subjects, seed=42)
    assert set(mapping.keys()) == set(subjects)
    assert set(mapping.values()) == {"train", "val", "test"}


def test_unseen_classes_never_in_train_or_val():
    labels, subjects = _label_subject_fixture(n_per_class=32, n_subjects=8, seed=7)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        record = write_splits_from_arrays(out_dir, labels=labels, subjects=subjects, seed=7)
        assert_no_unseen_leakage(record, labels)
        for name, info in record["splits"].items():
            per_window = np.array(info["per_window"], dtype=object)
            unseen = set(info["unseen_labels"])
            # Every unseen window is in either test or support_pool.
            for idx in np.where(np.isin(labels, list(unseen)))[0]:
                assert per_window[idx] in {"test", "support_pool"}, (
                    f"split {name} window {idx} label={labels[idx]} → {per_window[idx]}"
                )


def test_split_C_contains_dataset_unique_labels_only_in_test():
    labels, subjects = _label_subject_fixture(n_per_class=16, n_subjects=8, seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        rec = write_splits_from_arrays(out_dir, labels=labels, subjects=subjects, seed=11)
        c = rec["splits"]["C"]
        unseen = set(c["unseen_labels"])
        assert {"iron", "nordic_walk", "elevator_up", "elevator_down"}.issubset(unseen)
