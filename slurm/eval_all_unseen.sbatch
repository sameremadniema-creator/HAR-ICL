#!/bin/bash
#SBATCH --job-name=har_eval_unseen
#SBATCH --partition=3090
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/evalunseen_%A_%a.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/evalunseen_%A_%a.err


set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
REPO_DIR=${HAR_REPO:-$ROOT/har_icl}
RANK=${RANK:-16}
SEEDS=(42 1337 2024 7 31337 99 2025 31415)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
conda activate "$ENV_PREFIX" || { echo "env missing"; exit 1; }
cd "$REPO_DIR"
python scripts/assert_cuda.py || { echo "GPU not usable"; exit 1; }

TOK="$ROOT/tokens/vqvae_usc_t32_s${SEED}"
CK="$ROOT/checkpoints/gpt2/vqvae_usc_t32_splitC_k4_r${RANK}_s${SEED}_matched"

echo "===== all-unseen eval seed ${SEED} ====="
python -m phase4_eval.eval --ckpt "$CK/best" --tokens "$TOK" --split C --k_shot 4 --seed "$SEED" \
  --distractor_mode all_unseen --out "$CK/eval_all_unseen.json"
grep -E "unseen_macro_f1|unseen_acc|\"chance\"" "$CK/eval_all_unseen.json" | head
echo "[done] all-unseen seed ${SEED}"
