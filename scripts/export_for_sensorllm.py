"""Export harmonised windows to a JSONL SensorLLM can ingest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="C")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    data = np.load(Path(args.data) / "windows.npz", allow_pickle=True)
    splits = json.loads((Path(args.data) / "splits.json").read_text())
    per_window = np.array(splits["splits"][args.split]["per_window"], dtype=object)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for subset in ("train", "val", "test"):
        mask = per_window == subset
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        path = out_dir / f"{subset}.jsonl"
        with open(path, "w") as f:
            for i in idx:
                rec = {
                    "window": data["X"][i].tolist(),
                    "label": str(data["y"][i]),
                    "subject": str(data["subjects"][i]),
                    "dataset": str(data["datasets"][i]),
                    "split": subset,
                }
                f.write(json.dumps(rec) + "\n")
        counts[subset] = int(len(idx))
    print(json.dumps({"out": str(out_dir), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
