#!/usr/bin/env python3
"""Aggregate the matched multi-seed K-shot curve (K=1/2/4/8) across 8 seeds."""
from __future__ import annotations
import argparse, glob, json, os, statistics as st

SEEDS = [42, 1337, 2024, 7, 31337, 99, 2025, 31415]
KS = [1, 2, 4, 8]
RANK = int(os.environ.get("RANK", "16"))


def load(path):
    if not os.path.exists(path):
        return None
    try:
        r = json.load(open(path))
    except Exception:
        return None
    u = r.get("unseen_macro_f1", r.get("test_unseen_f1"))
    s = r.get("seen_macro_f1", r.get("test_seen_f1"))
    return None if u is None else (float(u), float(s) if s is not None else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("HAR_ROOT", "/mnt/fast/nobackup/users/aa08742"))
    args = ap.parse_args()
    R = args.root
    print(f"{'K':>3} {'n':>3} {'unseen mean±std':>20} {'seen mean':>10}   per-seed unseen")
    for K in KS:
        vals, seens = [], []
        for s in SEEDS:
            for g in (f"{R}/checkpoints/gpt2/vqvae_usc_t32_splitC_k{K}_r{RANK}_s{s}_matched/eval.json",
                      f"{R}/**/vqvae_usc_t32_splitC_k{K}_r{RANK}_s{s}_matched/eval.json"):
                hit = None
                for p in glob.glob(g, recursive=True):
                    hit = load(p)
                    if hit:
                        break
                if hit:
                    vals.append(hit[0]); seens.append(hit[1]); break
        if vals:
            m, sd = st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)
            print(f"{K:>3} {len(vals):>3} {m:>10.3f} ± {sd:<7.3f} {st.mean(seens):>10.3f}   {[round(v,3) for v in vals]}")
        else:
            print(f"{K:>3} {0:>3} {'—':>20} {'—':>10}   (no eval.json yet)")


if __name__ == "__main__":
    main()
