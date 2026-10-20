#!/usr/bin/env python3
"""Standalone CLI to fetch (or check the presence of) HAR datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `har_icl` importable when running this file directly via `python.
_REPO = Path(__file__).resolve().parent.parent
_PARENT = _REPO.parent
for p in (str(_REPO), str(_PARENT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from phase0_data.download import (  # noqa: E402
    DATA_URLS,
    download_all,
    status_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download HAR datasets (UCI-HAR, PAMAP2, MHealth, USC-HAD) to a local "
            "directory. Idempotent: skips datasets that are already present."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for usage examples.",
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Where to write raw data subdirectories (default: data/raw).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATA_URLS),
        choices=list(DATA_URLS),
        help=f"Subset to fetch. Default: all ({', '.join(DATA_URLS)}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if data is already present on disk.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Report what's on disk; do not perform any downloads.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON only (no human-readable summary).",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.status:
        rows = status_report(raw_dir)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        _print_status(rows, raw_dir)
        return 0

    results = download_all(
        raw_dir, datasets=args.datasets, force=args.force, verbose=not args.json
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_results(results, raw_dir)
    # Exit non-zero if anything genuinely failed (vs manual-required, which is.
    return 0 if not any(r["status"] == "failed" for r in results) else 2


def _print_status(rows, raw_dir: Path) -> None:
    print(f"Dataset status under {raw_dir}/")
    print(f"  {'dataset':<12} {'present':<9}  reason")
    print(f"  {'-'*12} {'-'*9}  {'-'*40}")
    for r in rows:
        flag = "yes" if r["present"] else "no"
        print(f"  {r['name']:<12} {flag:<9}  {r['reason']}")


def _print_results(results, raw_dir: Path) -> None:
    print(f"\nResults under {raw_dir}/")
    print(f"  {'dataset':<12} {'status':<22}  details")
    print(f"  {'-'*12} {'-'*22}  {'-'*48}")
    for r in results:
        detail = r.get("reason") or r.get("error") or r.get("path", "")
        print(f"  {r['name']:<12} {r['status']:<22}  {detail}")
    n_present = sum(1 for r in results if r["status"] == "present")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_manual = sum(1 for r in results if r["status"] == "manual_required")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(
        f"\nSummary: {n_present} already present · {n_ok} downloaded · "
        f"{n_manual} manual-required · {n_failed} failed"
    )
    if n_manual or n_failed:
        print("\nFollow the printed instructions for any dataset marked manual_required/failed.")


if __name__ == "__main__":
    sys.exit(main())
