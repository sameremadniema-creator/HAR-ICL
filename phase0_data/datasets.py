"""Dataset-specific loaders → (windows, raw_labels, subjects) at 50 Hz."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _window_stream(
    signal: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    length: int = 128,
    stride: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window a continuous (T, C) signal into (N, length, C) windows."""
    if signal.ndim != 2:
        raise ValueError(f"expected (T, C), got {signal.shape}")
    T, C = signal.shape
    if T < length:
        return (
            np.empty((0, length, C), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int32),
        )
    out_X, out_y, out_s = [], [], []
    for start in range(0, T - length + 1, stride):
        end = start + length
        seg_y = labels[start:end]
        # Pure-label window only.
        if not np.all(seg_y == seg_y[0]):
            continue
        out_X.append(signal[start:end].astype(np.float32))
        out_y.append(seg_y[0])
        out_s.append(int(subject_ids[start]))
    if not out_X:
        return (
            np.empty((0, length, C), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.stack(out_X, axis=0).astype(np.float32),
        np.array(out_y, dtype=object),
        np.array(out_s, dtype=np.int32),
    )


def _decimate_50hz(x: np.ndarray, from_hz: float) -> np.ndarray:
    """Anti-aliased downsample to 50 Hz along axis 0."""
    if abs(from_hz - 50.0) < 1e-6:
        return x
    from scipy.signal import decimate

    q = int(round(from_hz / 50.0))
    if q <= 1:
        return x
    return decimate(x, q=q, axis=0, ftype="iir", zero_phase=True).astype(np.float32)


# ─────────────────────── UCI-HAR (50 Hz, pre-windowed) ───────────────────────

UCI_INERTIAL_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]


UCI_LABEL_MAP = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}


def load_uci_har(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two paths"""
    root = Path(root)
    inertial = None
    for cand in (root, root / "UCI HAR Dataset", root / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"):
        if (cand / "train" / "Inertial Signals").exists():
            inertial = cand
            break
    if inertial is None:
        # Try ucimlrepo path — only features are available there, raise informatively.
        if (root / "X.csv").exists():
            raise RuntimeError(
                "UCI-HAR was fetched via ucimlrepo which provides only the 561 hand-crafted "
                "features, NOT the raw IMU signals required by this pipeline. Please download "
                "the full archive (https://archive.ics.uci.edu/static/public/240/...) and unzip "
                f"into {root}/"
            )
        raise FileNotFoundError(f"UCI-HAR Inertial Signals not found under {root}")

    def _read_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sig_dir = inertial / split / "Inertial Signals"
        chans = []
        for cname in UCI_INERTIAL_CHANNELS:
            fpath = sig_dir / f"{cname}_{split}.txt"
            arr = np.loadtxt(fpath, dtype=np.float32)  # (N, 128)
            chans.append(arr)
        windows = np.stack(chans, axis=-1)  # (N, 128, 6)
        y_int = np.loadtxt(inertial / split / f"y_{split}.txt", dtype=int)
        labels = np.array([UCI_LABEL_MAP[i] for i in y_int], dtype=object)
        subjects = np.loadtxt(inertial / split / f"subject_{split}.txt", dtype=int).astype(np.int32)
        return windows.astype(np.float32), labels, subjects

    X_train, y_train, s_train = _read_split("train")
    X_test, y_test, s_test = _read_split("test")
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    s = np.concatenate([s_train, s_test], axis=0)
    return X, y, s


# ─────────────────────── PAMAP2 (100 Hz IMU, decimate to 50 Hz) ───────────────

PAMAP2_ACTIVITY_MAP = {
    1: "lying", 2: "sitting", 3: "standing",
    4: "walking", 5: "running", 6: "cycling",
    7: "nordic walking", 9: "watching TV", 10: "computer work",
    11: "car driving", 12: "ascending stairs", 13: "descending stairs",
    16: "vacuum cleaning", 17: "ironing", 18: "folding laundry",
    19: "house cleaning", 20: "playing soccer", 24: "rope jumping",
    0: "transient activity",
}


def _find_pamap2_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (root, root / "PAMAP2_Dataset"):
        for sub in ("Protocol", "Optional"):
            d = base / sub
            if d.exists():
                candidates.extend(sorted(d.glob("subject1*.dat")))
    return candidates


def load_pamap2(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PAMAP2 .dat columns (54 total): timestamp, activityID, HR, then 3 IMUs of"""
    root = Path(root)
    files = _find_pamap2_files(root)
    if not files:
        raise FileNotFoundError(f"No PAMAP2 .dat files under {root}")
    # PAMAP2 wrist IMU column indices (0-indexed)
    HAND_ACC = [4, 5, 6]
    HAND_GYRO = [10, 11, 12]

    out_X: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    out_s: list[np.ndarray] = []
    for fpath in files:
        try:
            data = np.loadtxt(fpath, dtype=np.float32)
        except Exception:
            continue
        if data.size == 0 or data.ndim != 2:
            continue
        # Subject ID from filename: subject10X.dat → X
        stem = fpath.stem  # e.g. subject101
        try:
            sid = int(stem.replace("subject", ""))
        except ValueError:
            sid = -1
        activity = data[:, 1].astype(int)
        sig = data[:, HAND_ACC + HAND_GYRO]
        # Replace NaNs with column means.
        if np.isnan(sig).any():
            col_means = np.nanmean(sig, axis=0)
            inds = np.where(np.isnan(sig))
            sig[inds] = np.take(col_means, inds[1])
        # Map activity IDs to label strings; unknown → "transient activity".
        labels = np.array(
            [PAMAP2_ACTIVITY_MAP.get(int(a), "transient activity") for a in activity],
            dtype=object,
        )
        subjects = np.full(len(sig), sid, dtype=np.int32)
        # Decimate 100Hz → 50Hz (signals only, then realign labels/subjects by index).
        sig_50 = _decimate_50hz(sig, 100.0)
        # Match labels to new length by simple stride 2 (labels are slowly varying).
        idx = np.arange(0, len(labels), 2)[: len(sig_50)]
        labels_50 = labels[idx]
        subjects_50 = subjects[idx]
        X_w, y_w, s_w = _window_stream(sig_50, labels_50, subjects_50)
        if len(X_w):
            out_X.append(X_w)
            out_y.append(y_w)
            out_s.append(s_w)
    if not out_X:
        return (
            np.empty((0, 128, 6), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.concatenate(out_X, axis=0),
        np.concatenate(out_y, axis=0),
        np.concatenate(out_s, axis=0),
    )


# ──────────────────────────── MHealth (50 Hz) ────────────────────────────────

MHEALTH_LABEL_MAP = {
    0: "null", 1: "Standing still", 2: "Sitting and relaxing", 3: "Lying down",
    4: "Walking", 5: "Climbing stairs", 6: "Waist bends forward",
    7: "Frontal elevation of arms", 8: "Knees bending (crouching)",
    9: "Cycling", 10: "Jogging", 11: "Running",
    12: "Jump front and back",
}


def load_mhealth(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MHealth columns (24 total)"""
    root = Path(root)
    files = sorted(root.rglob("mHealth_subject*.log"))
    if not files:
        raise FileNotFoundError(f"No MHealth_subject*.log under {root}")
    WRIST_ACC = [14, 15, 16]
    WRIST_GYRO = [17, 18, 19]

    out_X: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    out_s: list[np.ndarray] = []
    for fpath in files:
        data = np.loadtxt(fpath, dtype=np.float32)
        if data.size == 0 or data.shape[1] < 24:
            continue
        stem = fpath.stem  # mHealth_subject1
        try:
            sid = int(stem.replace("mHealth_subject", ""))
        except ValueError:
            sid = -1
        activity = data[:, 23].astype(int)
        sig = data[:, WRIST_ACC + WRIST_GYRO]
        labels = np.array(
            [MHEALTH_LABEL_MAP.get(int(a), "null") for a in activity],
            dtype=object,
        )
        subjects = np.full(len(sig), sid, dtype=np.int32)
        X_w, y_w, s_w = _window_stream(sig, labels, subjects)
        if len(X_w):
            out_X.append(X_w)
            out_y.append(y_w)
            out_s.append(s_w)
    if not out_X:
        return (
            np.empty((0, 128, 6), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.concatenate(out_X, axis=0),
        np.concatenate(out_y, axis=0),
        np.concatenate(out_s, axis=0),
    )


# ───────────────── USC-HAD (100 Hz right-hip → 50 Hz) ────────────────────────

USC_HAD_ACTIVITIES = {
    1: "walking forward", 2: "walking left", 3: "walking right",
    4: "walking upstairs", 5: "walking downstairs", 6: "running forward",
    7: "jumping up", 8: "sitting", 9: "standing", 10: "sleeping",
    11: "elevator up", 12: "elevator down",
}


def load_usc_had(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """USC-HAD ships per-trial .mat files at sipi.usc.edu/had with structure"""
    from scipy.io import loadmat

    root = Path(root)
    mat_files = sorted(root.rglob("a*t*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No USC-HAD a*t*.mat files under {root}")

    out_X: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    out_s: list[np.ndarray] = []
    for fpath in mat_files:
        # path .../Subject{N}/a{ACT}t{TRIAL}.mat
        parts = fpath.parts
        subj_dir = next((p for p in parts if p.startswith("Subject")), None)
        if subj_dir is None:
            continue
        try:
            sid = int(subj_dir.replace("Subject", ""))
        except ValueError:
            sid = -1
        stem = fpath.stem  # a3t2
        try:
            aid = int(stem.split("t")[0].lstrip("a"))
        except ValueError:
            continue
        m = loadmat(str(fpath))
        if "sensor_readings" not in m:
            continue
        sig = np.asarray(m["sensor_readings"], dtype=np.float32)
        if sig.ndim != 2 or sig.shape[1] < 6:
            continue
        sig = sig[:, :6]
        labels = np.full(len(sig), USC_HAD_ACTIVITIES.get(aid, "unknown"), dtype=object)
        subjects = np.full(len(sig), sid, dtype=np.int32)
        sig_50 = _decimate_50hz(sig, 100.0)
        idx = np.arange(0, len(labels), 2)[: len(sig_50)]
        X_w, y_w, s_w = _window_stream(sig_50, labels[idx], subjects[idx])
        if len(X_w):
            out_X.append(X_w)
            out_y.append(y_w)
            out_s.append(s_w)
    if not out_X:
        return (
            np.empty((0, 128, 6), dtype=np.float32),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=np.int32),
        )
    return (
        np.concatenate(out_X, axis=0),
        np.concatenate(out_y, axis=0),
        np.concatenate(out_s, axis=0),
    )


LOADERS = {
    "uci_har": load_uci_har,
    "pamap2": load_pamap2,
    "mhealth": load_mhealth,
    "usc_had": load_usc_had,
}
