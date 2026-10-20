"""Idempotent dataset downloaders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

# ─────────────────────── Per-dataset metadata ──────────────────────────────

DATA_URLS: dict[str, dict] = {
    "uci_har": {
        "kind": "ucimlrepo",
        "uci_id": 240,
        "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "manual": (
            "UCI-HAR auto-download (via ucimlrepo) provides only the 561 hand-crafted "
            "features. For the raw IMU signals we also need, download the FULL archive:\n"
            "  https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip\n"
            "and unzip into data/raw/uci_har/  (expect UCI HAR Dataset/train/Inertial Signals/)"
        ),
    },
    "pamap2": {
        "kind": "zip",
        "url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "manual": (
            "Download https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip\n"
            "and unzip into data/raw/pamap2/  (expect PAMAP2_Dataset/Protocol/subject101.dat ...)"
        ),
    },
    "mhealth": {
        "kind": "zip",
        "url": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
        "manual": (
            "Download https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip\n"
            "and unzip into data/raw/mhealth/  (expect mHealth_subject1.log ...)"
        ),
    },
    "usc_had": {
        "kind": "manual",
        "url": "https://sipi.usc.edu/had/USC-HAD.zip",
        "manual": (
            "USC-HAD is auth-walled. Visit https://sipi.usc.edu/had/  → request the archive →\n"
            "unzip into data/raw/usc_had/  (expect Subject1/a1t1.mat ...)"
        ),
    },
}


# ─────────────────────── Presence checks ──────────────────────────────────


def _check_uci_har(target: Path) -> tuple[bool, str]:
    """UCI-HAR present if either raw inertial signals OR ucimlrepo CSVs exist."""
    candidates = [
        target / "train" / "Inertial Signals",
        target / "UCI HAR Dataset" / "train" / "Inertial Signals",
        target / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset" / "train" / "Inertial Signals",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("body_acc_x_*.txt")):
            return True, f"raw inertial signals at {c}"
    csv_train_x = target / "X.csv"
    csv_train_y = target / "y.csv"
    if csv_train_x.exists() and csv_train_y.exists():
        return True, f"ucimlrepo CSVs at {target} (features-only — raw signals missing)"
    return False, f"no inertial signals or CSVs under {target}"


def _check_pamap2(target: Path) -> tuple[bool, str]:
    for base in (target, target / "PAMAP2_Dataset"):
        for sub in ("Protocol", "Optional"):
            d = base / sub
            if d.exists() and any(d.glob("subject1*.dat")):
                return True, f"{d} contains subject*.dat"
    return False, f"no subject*.dat under {target}"


def _check_mhealth(target: Path) -> tuple[bool, str]:
    matches = list(target.rglob("mHealth_subject*.log"))
    if matches:
        return True, f"{len(matches)} mHealth_subject*.log files under {target}"
    return False, f"no mHealth_subject*.log files under {target}"


def _check_usc_had(target: Path) -> tuple[bool, str]:
    matches = list(target.rglob("a*t*.mat"))
    if matches:
        return True, f"{len(matches)} a*t*.mat files under {target}"
    return False, f"no a*t*.mat files under {target}"


CHECKERS: dict[str, Callable[[Path], tuple[bool, str]]] = {
    "uci_har": _check_uci_har,
    "pamap2": _check_pamap2,
    "mhealth": _check_mhealth,
    "usc_had": _check_usc_had,
}


def is_present(name: str, raw_dir: Path) -> tuple[bool, str]:
    """Return (present, message) for a single dataset."""
    if name not in CHECKERS:
        raise KeyError(f"Unknown dataset: {name}. Known: {sorted(CHECKERS)}")
    target = raw_dir / name
    if not target.exists():
        return False, f"directory does not exist: {target}"
    return CHECKERS[name](target)


# ─────────────────────── Low-level helpers ─────────────────────────────────


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "har-icl/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def _fetch_uci_har(out_dir: Path, prefer_full_archive: bool = True) -> None:
    """Download UCI-HAR."""
    if prefer_full_archive:
        url = DATA_URLS["uci_har"]["url"]
        zip_path = out_dir / "_archive.zip"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            _download(url, zip_path)
        _unzip(zip_path, out_dir)
        # Inner ZIP: "UCI HAR Dataset.zip" → extract.
        for inner in list(out_dir.rglob("*.zip")):
            if inner == zip_path:
                continue
            try:
                _unzip(inner, inner.parent)
            except zipfile.BadZipFile:
                pass
        return
    # Fallback: ucimlrepo features-only.
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
    except ImportError:
        raise RuntimeError(
            "ucimlrepo not installed. `pip install ucimlrepo` or download UCI-HAR manually."
        )
    ds = fetch_ucirepo(id=240)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.data.features.to_csv(out_dir / "X.csv", index=False)
    ds.data.targets.to_csv(out_dir / "y.csv", index=False)
    print(f"[download] UCI-HAR -> {out_dir} (features+targets via ucimlrepo, no raw IMU)", flush=True)


def _fetch_zip(name: str, url: str, raw_dir: Path) -> None:
    target = raw_dir / name
    target.mkdir(parents=True, exist_ok=True)
    zip_path = target / "_archive.zip"
    if not zip_path.exists():
        _download(url, zip_path)
    _unzip(zip_path, target)
    # Recursively unzip any nested archives (PAMAP2 ships nested).
    for inner in list(target.rglob("*.zip")):
        if inner == zip_path:
            continue
        try:
            _unzip(inner, inner.parent)
        except zipfile.BadZipFile:
            pass


# ─────────────────────── Public API ────────────────────────────────────────


def download_dataset(
    name: str,
    raw_dir: Path,
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Download one dataset if not already present."""
    if name not in DATA_URLS:
        raise KeyError(f"Unknown dataset: {name}. Known: {sorted(DATA_URLS)}")
    info = DATA_URLS[name]
    target_dir = raw_dir / name

    present, reason = is_present(name, raw_dir)
    if present and not force:
        if verbose:
            print(f"[download] {name}: present ({reason}) — skipping.", flush=True)
        return {"name": name, "status": "present", "path": str(target_dir), "reason": reason}

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        kind = info["kind"]
        if kind == "ucimlrepo":
            _fetch_uci_har(target_dir, prefer_full_archive=True)
        elif kind == "zip":
            _fetch_zip(name, info["url"], raw_dir)
        elif kind == "manual":
            print(
                f"[download] {name}: auto-fetch not supported (auth-walled).\n"
                f"{info['manual']}",
                file=sys.stderr,
            )
            return {"name": name, "status": "manual_required", "path": str(target_dir)}
        else:
            raise ValueError(f"unknown kind: {kind}")
        # Verify after download.
        present_after, reason_after = is_present(name, raw_dir)
        return {
            "name": name,
            "status": "ok" if present_after else "downloaded_but_unverified",
            "path": str(target_dir),
            "verified": bool(present_after),
            "reason": reason_after,
        }
    except Exception as e:
        print(
            f"[download] {name} failed: {e}\nManual instructions:\n{info['manual']}",
            file=sys.stderr,
        )
        return {
            "name": name,
            "status": "failed",
            "error": str(e),
            "path": str(target_dir),
        }


def download_all(
    raw_dir: str | Path,
    *,
    datasets: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> list[dict]:
    raw_dir = Path(raw_dir)
    names = datasets if datasets is not None else list(DATA_URLS)
    return [download_dataset(n, raw_dir, force=force, verbose=verbose) for n in names]


def status_report(raw_dir: str | Path) -> list[dict]:
    """Lightweight "what's on disk?" report without any network access."""
    raw_dir = Path(raw_dir)
    rows = []
    for name in DATA_URLS:
        present, reason = is_present(name, raw_dir)
        rows.append({
            "name": name,
            "present": bool(present),
            "path": str(raw_dir / name),
            "reason": reason,
        })
    return rows


# ─────────────────────── CLI (backwards-compatible) ────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HAR datasets to data/raw/. Idempotent: skips datasets "
                    "already present unless --force is given."
    )
    parser.add_argument(
        "--raw-dir", type=str, default=os.environ.get("HAR_RAW_DIR", "data/raw")
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATA_URLS),
        help=f"Subset to download. Default: all ({', '.join(DATA_URLS)})",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument("--status", action="store_true",
                        help="Just report what's on disk; no downloads.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if args.status:
        print(json.dumps(status_report(raw_dir), indent=2))
        return
    results = download_all(raw_dir, datasets=args.datasets, force=args.force)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
