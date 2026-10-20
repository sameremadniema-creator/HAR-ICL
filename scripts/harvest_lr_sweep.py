#!/usr/bin/env python3
"""Harvest the continuous-control LR sweep: seen/unseen macro-F1 vs learning rate, over 8."""
from __future__ import annotations
import json, glob, os, re, statistics as st

R = os.environ.get("HAR_ROOT", "/mnt/fast/nobackup/users/aa08742") + "/checkpoints/baselines/continuous_projection"
SEEDS = [42, 1337, 2024, 7, 31337, 99, 2025, 31415]

def load(p):
    if not os.path.exists(p): return None
    try: r = json.load(open(p))
    except Exception: return None
    return r.get("unseen_macro_f1"), r.get("seen_macro_f1")

# discover LRs present (from _lr<LR>.json filenames) + the matched 2e-4 point
lrs = sorted({m.group(1) for f in glob.glob(f"{R}/splitC_k4_s*_lr*.json")
              for m in [re.search(r"_lr([0-9.eE+-]+)\.json$", f)] if m},
             key=lambda x: float(x))
rows = [("2e-4 (matched)", lambda s: f"{R}/splitC_k4_s{s}_matched.json")]
rows += [(lr, (lambda s, lr=lr: f"{R}/splitC_k4_s{s}_lr{lr}.json")) for lr in lrs]

print(f"{'learning rate':>16} {'n':>3} {'unseen-F1 mean±sd':>20} {'seen-F1 mean±sd':>20}")
print("-"*64)
for label, pathf in rows:
    uu, ss = [], []
    for s in SEEDS:
        v = load(pathf(s))
        if v and v[0] is not None: uu.append(v[0]); ss.append(v[1] if v[1] is not None else float('nan'))
    if not uu: print(f"{label:>16}  (no files)"); continue
    um, us = st.mean(uu), (st.pstdev(uu) if len(uu) > 1 else 0.0)
    sm, ss2 = st.mean(ss), (st.pstdev(ss) if len(ss) > 1 else 0.0)
    print(f"{label:>16} {len(uu):>3}   {um:.3f} ± {us:<7.3f}   {sm:.3f} ± {ss2:<7.3f}")
print("\nchance ≈ 0.249 (unseen-F1) / 0.146 (seen-F1). Discrete headline = 0.478 unseen / 0.941 seen.")
print("If seen-F1 stays ≈0.15 across LRs → the continuous bridge never acquires in-context copying at any LR.")
