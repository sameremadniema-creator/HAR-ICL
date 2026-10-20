#!/bin/bash
#SBATCH --job-name=har_confound
#SBATCH --partition=3090
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/confound_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/confound_%j.err

# DEVICE-LEAKAGE / CONFOUND ABLATION (regenerates the deleted analysis), on the.

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
REPO_DIR=${HAR_REPO:-$ROOT/har_icl}
SEED=${SEED:-42}
VQ_CFG=${VQ_CFG:-vqvae_aux_contrast_t32}

mkdir -p "$REPO_DIR"/logs/slurm "$ROOT"/checkpoints/vqvae "$ROOT"/tokens
source ~/.bashrc
conda activate "$ENV_PREFIX" || { echo "env missing at $ENV_PREFIX"; exit 1; }
cd "$REPO_DIR"
python scripts/assert_cuda.py || { echo "Aborting: GPU not usable." >&2; exit 1; }

DX="$REPO_DIR/data/processed/v1"          # full cross-dataset real data (has the 'datasets' field)
CX="$ROOT/checkpoints/vqvae/${VQ_CFG}_s${SEED}"
TX="$ROOT/tokens/${VQ_CFG}_s${SEED}"

echo "===== train cross-dataset tokenizer (if missing) ====="
[ -f "$CX/best.pt" ] || python -m phase1_vqvae.train --config configs/vqvae/${VQ_CFG}.yaml \
  --data "$DX" --seed "$SEED" --split C --ckpt_dir "$CX"
echo "===== tokenize cross-dataset (if missing) ====="
[ -f "$TX/split_C/test.npz" ] || python -m phase2_tokenize.encode_signals \
  --vqvae_ckpt "$CX/best.pt" --data "$DX" --out "$TX" --splits C

echo "===== confound controls ====="
for R in "crossdataset:" "pamap2:--restrict_dataset pamap2" "usc:--restrict_dataset usc_had"; do
  nm=${R%%:*}; fl=${R#*:}
  echo "----- recovery [$nm] -----"
  python -m phase4_eval.codebook_recovery --tokens "$TX" --data "$DX" --ckpt "$CX/best.pt" \
    --split C --k_shot 4 --episode_reps 5 $fl --out "$TX/recovery_$nm.json" || echo "[WARN] $nm failed"
done

echo "############### CONFOUND SUMMARY ###############"
grep -E "name_that_dataset_acc|dataset_only|\"hist\"|\"zq_mean\"|\"z_mean\"|\"random\"" \
  "$TX"/recovery_crossdataset.json 2>/dev/null | head -30 || true
echo "-- device-constant pamap2 --"; grep -E "\"hist\"|\"zq_mean\"|\"random\"" "$TX"/recovery_pamap2.json 2>/dev/null | head
echo "-- device-constant usc --";    grep -E "\"hist\"|\"zq_mean\"|\"random\"" "$TX"/recovery_usc.json 2>/dev/null | head
echo "[done] confound"
