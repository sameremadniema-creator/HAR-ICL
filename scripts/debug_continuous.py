#!/usr/bin/env python3
"""debug_continuous.py — CPU end-to-end smoke test for the NEW paper runs BEFORE"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Make the repo root importable (so `baselines` / `har_icl` resolve when run as a.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

WINDOW = 128
CHANNELS = 6
SEEN = ["walk", "run", "sit"]
UNSEEN = ["elevator_up", "elevator_down"]


def make_tiny_dataset(d: Path, per_label: int = 10) -> None:
    """Write windows.npz / meta.json / splits.json in the exact shape load_split_arrays reads."""
    rng = np.random.default_rng(0)
    labels = SEEN + UNSEEN
    X, y, per_window = [], [], []
    for lbl in labels:
        for j in range(per_label):
            X.append(rng.standard_normal((WINDOW, CHANNELS)).astype(np.float32))
            y.append(lbl)
            if lbl in UNSEEN:
                # unseen classes only ever appear as support_pool (support) and test (queries)
                per_window.append("support_pool" if j % 2 == 0 else "test")
            else:
                per_window.append(["train", "train", "val", "test"][j % 4])
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=object)
    label_set = labels
    label_to_int = {l: i for i, l in enumerate(label_set)}
    np.savez(d / "windows.npz", X=X, y=y)
    (d / "meta.json").write_text(json.dumps({
        "label_set": label_set,
        "label_to_int": label_to_int,
        "label_descriptions": {l: l for l in label_set},
        "source": "smoke",
    }))
    (d / "splits.json").write_text(json.dumps({
        "splits": {"C": {"per_window": per_window}}
    }))


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_baseline(name: str, module: str, extra: list[str], data_dir: Path, out: Path) -> tuple[bool, str]:
    """Invoke a baseline EXACTLY like paper_baselines.sbatch does (python -m ...), with tiny"""
    import subprocess
    cmd = [sys.executable, "-m", module,
           "--data", str(data_dir), "--split", "C", "--seed", "0", "--out", str(out), *extra]
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        return False, f"exit {proc.returncode}: {tail}"
    if not out.exists():
        return False, "ran but wrote no output json"
    try:
        res = json.loads(out.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"bad json: {e}"
    if "overall_macro_f1" not in res:
        return False, f"json missing overall_macro_f1 (keys={list(res)[:6]})"
    have = {k: round(res[k], 3) for k in ("seen_macro_f1", "unseen_macro_f1", "overall_macro_f1") if k in res}
    return True, f"ran OK; metrics={have}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the tiny dataset dir")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="har_smoke_"))
    data_dir = tmp / "usc_smoke"
    data_dir.mkdir(parents=True)
    make_tiny_dataset(data_dir)

    # tiny epochs / k so each runs in seconds; same modules paper_baselines.sbatch.
    checks = [
        ("continuous_projection", "baselines.continuous_projection",
         ["--epochs", "1", "--k_shot", "2", "--lora_rank", "4"]),
        ("protonet", "baselines.protonet", ["--k_shot", "2"]),
        ("attend", "baselines.attend", ["--epochs", "1"]),
    ]

    ok = True
    print(f"{'baseline':24s} {'status':6s}  detail")
    for nm, mod, extra in checks:
        passed, detail = run_baseline(nm, mod, extra, data_dir, tmp / f"{nm}.json")
        ok = ok and passed
        print(f"{nm:24s} {'OK' if passed else 'FAIL':6s}  {detail}")

    if not args.keep:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nSMOKE PASSED — paper_baselines path runs end-to-end, safe to submit."
          if ok else "\nSMOKE FAILED — fix before submitting.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
