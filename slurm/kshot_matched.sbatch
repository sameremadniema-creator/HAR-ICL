#!/bin/bash
#SBATCH --job-name=har_kshot_m
#SBATCH --partition=3090
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/kshotm_%A_%a.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/kshotm_%A_%a.err


set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
REPO_DIR=${HAR_REPO:-$ROOT/har_icl}
SEEDS=(42 1337 2024 7 31337 99 2025 31415)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
GPT_CFG=${GPT_CFG:-gpt2_k4}
RANK=${RANK:-16}
K_LIST=${K_LIST:-"1 2 8"}
EPOCHS=${EPOCHS:-15}
PATIENCE=${PATIENCE:-15}
MAXEV=${MAXEV:-600}

mkdir -p "$REPO_DIR"/logs/slurm "$ROOT"/checkpoints/gpt2
source ~/.bashrc
conda activate "$ENV_PREFIX" || { echo "env missing at $ENV_PREFIX"; exit 1; }
cd "$REPO_DIR"
python scripts/assert_cuda.py || { echo "Aborting: GPU not usable." >&2; exit 1; }

TOK="$ROOT/tokens/vqvae_usc_t32_s${SEED}"
run(){ local nm=$1; shift; echo "===== ${nm} ====="; "$@" || echo "[WARN] ${nm} failed"; }

echo "############### K-SHOT (matched) seed ${SEED}  K={${K_LIST}} ###############"
[ -f "$TOK/split_C/test.npz" ] || { echo "[FATAL] tokens missing at $TOK"; exit 1; }

for K in $K_LIST; do
  CK="$ROOT/checkpoints/gpt2/vqvae_usc_t32_splitC_k${K}_r${RANK}_s${SEED}_matched"
  echo "----- K=$K -----"
  run train_k${K} python -m phase3_gpt2.train --config configs/gpt2/${GPT_CFG}.yaml \
    --tokens "$TOK" --split C --k_shot "$K" --lora_rank "$RANK" --seed "$SEED" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --max_eval_examples "$MAXEV" --out "$CK"
  run eval_k${K} python -m phase4_eval.eval --ckpt "$CK/best" --tokens "$TOK" --split C --k_shot "$K" --seed "$SEED" \
    --out "$CK/eval.json"
  grep -E "unseen_macro_f1|seen_macro_f1|unseen_acc" "$CK/eval.json" 2>/dev/null | head -4
done
echo "[done] kshot matched seed ${SEED}"
