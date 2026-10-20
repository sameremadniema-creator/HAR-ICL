#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f data/processed/v1/windows.npz ] || { echo "STOP: data/processed/v1/windows.npz missing (source data)"; exit 1; }
EX="--exclude=aisurrey11"

echo "Submitting HAR-ICL pipeline (source: data/processed/v1)"
MID=$(sbatch $EX --array=0-7%4 --export=ALL,VQ_BSZ=64,MAXEP=2000 --parsable slurm/matched_run.sbatch)
echo "  [1] matched (headline)        : $MID"
KID=$(sbatch $EX --dependency=afterok:$MID --parsable slurm/kshot_matched.sbatch)
echo "  [2] k-shot                    : $KID   (after $MID)"
EID=$(sbatch $EX --dependency=afterok:$MID --parsable slurm/eval_all_unseen.sbatch)
echo "  [3] all-unseen eval           : $EID   (after $MID)"
BID=$(sbatch $EX --dependency=afterok:$MID --parsable slurm/baselines_rerun.sbatch)
echo "  [4] baselines                 : $BID   (after $MID)"
SID=$(sbatch $EX --parsable slurm/codebook_sweep.sbatch)
echo "  [5] tokeniser sweep (Table II): $SID"
CID=$(sbatch $EX --parsable slurm/confound.sbatch)
echo "  [6] device confound (Table III): $CID"
echo
squeue -u aa08742
echo
echo "When everything shows COMPLETED, harvest:"
echo "  python scripts/harvest_matched.py        # discrete vs continuous, 8/8, Wilcoxon"
echo "  python scripts/harvest_kshot_matched.py  # K-shot curve"
