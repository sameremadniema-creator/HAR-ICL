#!/usr/bin/env python3
"""Harvest the MATCHED-protocol discrete-vs-continuous rerun (slurm/matched_run.sbatch)."""
from __future__ import annotations
import argparse, glob, json, os, statistics as st

SEEDS = [42, 1337, 2024, 7, 31337, 99, 2025, 31415]
RANK = int(os.environ.get("RANK", "16"))


def _load_unseen(path):
    if not os.path.exists(path):
        return None
    try:
        r = json.load(open(path))
    except Exception:
        return None
    # discrete eval.json and continuous .json both expose these keys at top level
    u = r.get("unseen_macro_f1", r.get("test_unseen_f1"))
    s = r.get("seen_macro_f1", r.get("test_seen_f1"))
    return None if u is None else (float(u), float(s) if s is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("HAR_ROOT", "/mnt/fast/nobackup/users/aa08742"))
    args = ap.parse_args()
    R = args.root

    disc_glob = [f"{R}/checkpoints/gpt2/vqvae_usc_t32_splitC_k4_r{RANK}_s{{s}}_matched/eval.json",
                 f"{R}/**/vqvae_usc_t32_splitC_k4_r{RANK}_s{{s}}_matched/eval.json"]
    cont_glob = [f"{R}/checkpoints/baselines/continuous_projection/splitC_k4_s{{s}}_matched.json",
                 f"{R}/**/splitC_k4_s{{s}}_matched.json"]

    def find(globs, seed):
        for g in globs:
            for p in glob.glob(g.format(s=seed), recursive=True):
                v = _load_unseen(p)
                if v is not None:
                    return v, p
        return None, None

    disc, cont = {}, {}
    print(f"{'seed':>7} {'discrete(unseen/seen)':>26} {'continuous(unseen/seen)':>26}")
    for s in SEEDS:
        dv, dp = find(disc_glob, s)
        cv, cp = find(cont_glob, s)
        if dv: disc[s] = dv
        if cv: cont[s] = cv
        ds = f"{dv[0]:.3f}/{dv[1] if dv[1] is not None else float('nan'):.3f}" if dv else "—"
        cs = f"{cv[0]:.3f}/{cv[1] if cv[1] is not None else float('nan'):.3f}" if cv else "—"
        print(f"{s:>7} {ds:>26} {cs:>26}")

    du = [disc[s][0] for s in SEEDS if s in disc]
    cu = [cont[s][0] for s in SEEDS if s in cont]
    if du:
        print(f"\nDISCRETE   unseen-F1  mean={st.mean(du):.4f}  std={st.pstdev(du):.4f}  n={len(du)}")
    if cu:
        print(f"CONTINUOUS unseen-F1  mean={st.mean(cu):.4f}  std={st.pstdev(cu):.4f}  n={len(cu)}")

    common = [s for s in SEEDS if s in disc and s in cont]
    if len(common) >= 5:
        pa = [disc[s][0] for s in common]; pb = [cont[s][0] for s in common]
        wins = sum(a > b for a, b in zip(pa, pb))
        print(f"\npaired seeds n={len(common)}  discrete wins {wins}/{len(common)}")
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(pa, pb)
            print(f"Wilcoxon signed-rank  stat={stat}  p={p:.4f}")
        except Exception as e:
            print("scipy unavailable for Wilcoxon:", e)
    else:
        print(f"\n[note] only {len(common)} paired seeds found — rerun may still be in progress.")


if __name__ == "__main__":
    main()
