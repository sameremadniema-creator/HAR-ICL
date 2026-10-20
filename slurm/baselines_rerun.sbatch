#!/bin/bash
#SBATCH --job-name=har_base
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --exclude=aisurrey11
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/base_%A_%a.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/base_%A_%a.err

# Re-run ONLY the non-LLM baselines (ProtoNet, few-shot fine-tune, fixed-head.

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
REPO_DIR=${HAR_REPO:-$ROOT/har_icl}
SEEDS=(42 1337 2024 7 31337 99 2025 31415)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
conda activate "$ENV_PREFIX" || { echo "env missing"; exit 1; }
cd "$REPO_DIR"
python scripts/assert_cuda.py || { echo "GPU not usable" >&2; exit 1; }

PROC="$REPO_DIR/data/processed/usc_only_s${SEED}"      # the FIXED (train-only z-norm) device-free data
B="$ROOT/checkpoints/baselines"
mkdir -p "$B"/{protonet,simple_finetune,attend,deepconvlstm,deepconvlstm_attn,chronos_mlp}
[ -f "$PROC/windows.npz" ] || { echo "[FATAL] missing $PROC/windows.npz"; exit 1; }
run(){ local nm=$1; shift; echo "===== ${nm} ====="; "$@" || echo "[WARN] ${nm} failed"; }

echo "############### BASELINES (new z-norm) seed ${SEED} ###############"
run protonet          python -m baselines.protonet          --data "$PROC" --split C --seed "$SEED" --k_shot 4 --out "$B/protonet/splitC_k4_s${SEED}.json"
run simple_finetune   python -m baselines.simple_finetune   --data "$PROC" --split C --seed "$SEED" --k_shot 4 --out "$B/simple_finetune/splitC_k4_s${SEED}.json"
run attend            python -m baselines.attend            --data "$PROC" --split C --seed "$SEED" --out "$B/attend/splitC_s${SEED}.json"
run deepconvlstm      python -m baselines.deepconvlstm      --data "$PROC" --split C --seed "$SEED" --out "$B/deepconvlstm/splitC_s${SEED}.json"
run deepconvlstm_attn python -m baselines.deepconvlstm_attn --data "$PROC" --split C --seed "$SEED" --out "$B/deepconvlstm_attn/splitC_s${SEED}.json"
run chronos_mlp       python -m baselines.chronos_mlp       --data "$PROC" --split C --seed "$SEED" --out "$B/chronos_mlp/splitC_s${SEED}.json"
echo "[done] baselines seed ${SEED}"
