#!/bin/bash
#SBATCH --job-name=har_matched
#SBATCH --partition=3090
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/matched_%A_%a.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/matched_%A_%a.err


set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
REPO_DIR=${HAR_REPO:-$ROOT/har_icl}
SEEDS=(42 1337 2024 7 31337 99 2025 31415)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
UNSEEN=${UNSEEN:-elevator_up,elevator_down,walk_upstairs,walk_downstairs}
VQ_CFG=${VQ_CFG:-vqvae_aux_contrast_t32}
GPT_CFG=${GPT_CFG:-gpt2_k4}
RANK=${RANK:-16}
# Matched training protocol — same EPOCHS / PATIENCE / LR / val-metric for both.
EPOCHS=${EPOCHS:-20}
PATIENCE=${PATIENCE:-15}
LR=${LR:-2e-4}
WD=${WD:-0.0}
ACCUM=${ACCUM:-32}
MAXEP=${MAXEP:-0}
MAXEV=${MAXEV:-600}
VQ_BSZ=${VQ_BSZ:-128}
# STAGE = both | discrete | continuous → lets you run the two readers as SEPARATE.
STAGE=${STAGE:-both}

mkdir -p "$REPO_DIR"/logs/slurm "$ROOT"/checkpoints/gpt2 "$ROOT"/checkpoints/baselines/continuous_projection
source ~/.bashrc
conda activate "$ENV_PREFIX" || { echo "env missing at $ENV_PREFIX"; exit 1; }
cd "$REPO_DIR"
python scripts/assert_cuda.py || { echo "Aborting: GPU not usable." >&2; exit 1; }

PROC="$REPO_DIR/data/processed/usc_only_s${SEED}"
SRC="$REPO_DIR/data/processed/v1"
CK_VQ="$ROOT/checkpoints/vqvae/vqvae_usc_t32_s${SEED}"
TOK="$ROOT/tokens/vqvae_usc_t32_s${SEED}"
CK_GPT_M="$ROOT/checkpoints/gpt2/vqvae_usc_t32_splitC_k4_r${RANK}_s${SEED}_matched"
B="$ROOT/checkpoints/baselines"
run(){ local nm=$1; shift; echo "===== ${nm} ====="; "$@" || echo "[WARN] ${nm} failed"; }

echo "############### MATCHED seed ${SEED}  (epochs=${EPOCHS} patience=${PATIENCE} lr=${LR}) ###############"
# Reuse deterministic data/tokeniser/tokens (rebuild only if missing).
[ -f "$PROC/windows.npz" ] || python scripts/make_usc_device_free.py --data "$SRC" --out "$PROC" \
  --dataset usc_had --unseen "$UNSEEN" --seed "$SEED"
[ -f "$CK_VQ/best.pt" ] || python -m phase1_vqvae.train --config configs/vqvae/${VQ_CFG}.yaml \
  --data "$PROC" --seed "$SEED" --split C --ckpt_dir "$CK_VQ" --batch_size "$VQ_BSZ"
[ -f "$TOK/split_C/test.npz" ] || python -m phase2_tokenize.encode_signals \
  --vqvae_ckpt "$CK_VQ/best.pt" --data "$PROC" --out "$TOK" --splits C

# DISCRETE reader — matched protocol (retrain fresh into _matched).
if [[ "$STAGE" == "both" || "$STAGE" == "discrete" ]]; then
  run discrete_train python -m phase3_gpt2.train --config configs/gpt2/${GPT_CFG}.yaml \
    --tokens "$TOK" --split C --k_shot 4 --lora_rank "$RANK" --seed "$SEED" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --max_eval_examples "$MAXEV" --out "$CK_GPT_M"
  run discrete_eval python -m phase4_eval.eval --ckpt "$CK_GPT_M/best" --tokens "$TOK" --split C --k_shot 4 \
    --seed "$SEED" --out "$CK_GPT_M/eval.json"
  [ -f "$CK_GPT_M/eval.json" ] || { echo "[FATAL] matched discrete eval.json missing seed ${SEED}" >&2; exit 1; }
fi

# CONTINUOUS control — same epochs / patience / lr / train-cap / val-cap.
if [[ "$STAGE" == "both" || "$STAGE" == "continuous" ]]; then
  run continuous python -m baselines.continuous_projection --data "$PROC" --split C --seed "$SEED" \
    --vqvae_ckpt "$CK_VQ/best.pt" --epochs "$EPOCHS" --patience "$PATIENCE" \
    --max_train_episodes "$MAXEP" --max_eval_episodes "$MAXEV" --k_shot 4 --lora_rank "$RANK" \
    --lr "$LR" --weight_decay "$WD" --grad_accum "$ACCUM" \
    --out "$B/continuous_projection/splitC_k4_s${SEED}_matched.json"
fi

echo "===== SUMMARY (matched) seed ${SEED} ====="
grep -E "unseen_macro_f1|seen_macro_f1|overall_macro_f1|\"chance\"|unseen_acc" "$CK_GPT_M/eval.json" 2>/dev/null | head
echo "-- continuous --"; grep -E "unseen_macro_f1|seen_macro_f1" "$B/continuous_projection/splitC_k4_s${SEED}_matched.json" 2>/dev/null | head
echo "[done] matched seed ${SEED}"
