#!/bin/bash
#SBATCH --job-name=har_clr
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-7%4
#SBATCH --exclude=aisurrey11
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/clr_%A_%a.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/clr_%A_%a.err

# CONTINUOUS-CONTROL LEARNING-RATE SWEEP (robustness for the.

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
REPO_DIR=${HAR_REPO:-$ROOT/har_icl}
SEEDS=(42 1337 2024 7 31337 99 2025 31415)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RANK=${RANK:-16}
LR_LIST=${LR_LIST:-"5e-5 1e-4"}     # LRs chosen for the continuous arm (below the matched 2e-4)
EPOCHS=${EPOCHS:-20}; PATIENCE=${PATIENCE:-15}; MAXEP=${MAXEP:-2000}; MAXEV=${MAXEV:-600}

source ~/.bashrc
conda activate "$ENV_PREFIX" || { echo "env missing"; exit 1; }
cd "$REPO_DIR"
python scripts/assert_cuda.py || { echo "GPU not usable" >&2; exit 1; }

PROC="$REPO_DIR/data/processed/usc_only_s${SEED}"
CK_VQ="$ROOT/checkpoints/vqvae/vqvae_usc_t32_s${SEED}"
B="$ROOT/checkpoints/baselines"
[ -f "$PROC/windows.npz" ] || { echo "[FATAL] missing $PROC"; exit 1; }
[ -f "$CK_VQ/best.pt" ]    || { echo "[FATAL] missing $CK_VQ/best.pt"; exit 1; }
run(){ local nm=$1; shift; echo "===== ${nm} ====="; "$@" || echo "[WARN] ${nm} failed"; }

echo "############### CONTINUOUS LR SWEEP seed ${SEED}  LRs={${LR_LIST}} ###############"
for LR in $LR_LIST; do
  run "continuous lr=$LR" python -m baselines.continuous_projection --data "$PROC" --split C --seed "$SEED" \
    --vqvae_ckpt "$CK_VQ/best.pt" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --max_train_episodes "$MAXEP" --max_eval_episodes "$MAXEV" --k_shot 4 --lora_rank "$RANK" \
    --lr "$LR" --weight_decay 0.0 --grad_accum 32 \
    --out "$B/continuous_projection/splitC_k4_s${SEED}_lr${LR}.json"
  grep -E "\"seen_macro_f1\"|\"unseen_macro_f1\"|best_val_blended" "$B/continuous_projection/splitC_k4_s${SEED}_lr${LR}.json" 2>/dev/null | head -3
done
echo "[done] continuous LR sweep seed ${SEED}"
