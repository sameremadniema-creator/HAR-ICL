#!/bin/bash
#SBATCH --job-name=har_cb_sweep
#SBATCH --partition=3090
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=56G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/cbsweep_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/cbsweep_%j.err

# Sweep codebook designs and measure the UNSEEN prototype-F1 upper bound for each

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
REPO_DIR=${HAR_REPO:-$ROOT/har_icl}
mkdir -p "$REPO_DIR"/logs/slurm "$ROOT"/checkpoints/vqvae "$ROOT"/tokens "$REPO_DIR"/manifests

source ~/.bashrc
conda activate "$ENV_PREFIX" || { echo "env missing"; exit 1; }
cd "$REPO_DIR"
# Self-heal off broken GPU nodes: if CUDA isn't usable, exclude THIS node and.
if ! python scripts/assert_cuda.py; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "[guard] $(hostname) has no usable GPU — excluding it and requeuing job $SLURM_JOB_ID"
    scontrol update JobId="$SLURM_JOB_ID" ExcludeNodes="$(hostname)" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID" 2>/dev/null && { sleep 10; exit 0; }
  fi
  echo "Aborting: GPU not usable." >&2; exit 1
fi

PROC="$ROOT/data/processed/v1"
SEED=${SEED:-42}
SPLIT=${SPLIT:-C}
# Structural variants vs the previous winner aux_only (unseen 0.223, already on.
EXPS=${EXPS:-"vqvae_supcon_t32_tval vqvae_aux_contrast_t32"}

for EXP in $EXPS; do
  echo "================= $EXP ================="
  CKPT_DIR="$ROOT/checkpoints/vqvae/${EXP}_s${SEED}"
  TOK_DIR="$ROOT/tokens/${EXP}_s${SEED}"
  # Skip phase-1 retrain if a checkpoint already exists (set FORCE_TRAIN=1 to.
  if [ -f "$CKPT_DIR/best.pt" ] && [ "${FORCE_TRAIN:-0}" != "1" ]; then
    echo "[skip-train] $CKPT_DIR/best.pt exists — reusing it (FORCE_TRAIN=1 to retrain)."
  else
    python -m phase1_vqvae.train \
        --config configs/vqvae/${EXP}.yaml --data "$PROC" \
        --seed ${SEED} --split ${SPLIT} --ckpt_dir "$CKPT_DIR"
  fi
  # allow_conditioning: some ablation configs (cond_only, full_k*) tokenise WITH.
  python -m phase2_tokenize.encode_signals \
      --vqvae_ckpt "$CKPT_DIR/best.pt" --data "$PROC" \
      --out "$TOK_DIR" --splits A B C --allow_conditioning || { echo "[WARN] tokenize failed: $EXP"; continue; }
  python -m phase4_eval.codebook_recovery \
      --tokens "$TOK_DIR" --data "$PROC" --ckpt "$CKPT_DIR/best.pt" \
      --split ${SPLIT} --k_shot 4 --seed ${SEED} \
      --out "$TOK_DIR/codebook_recovery_split${SPLIT}.json" || { echo "[WARN] recovery failed: $EXP"; continue; }
done

echo; echo "================= SWEEP SUMMARY (unseen prototype-F1) ================="
python - <<PY
import json, glob, os
root="$ROOT"; seed="$SEED"; split="$SPLIT"
rows=[]
for d in sorted(glob.glob(f"{root}/tokens/*_s{seed}")):
    f=os.path.join(d, f"codebook_recovery_split{split}.json")
    if not os.path.exists(f): continue
    r=json.load(open(f))
    up=r["unseen_prototype_fewshot_macro_f1"]; sp=r["seen_linear_probe_macro_f1"]
    rows.append((os.path.basename(d), r["reconstruction"]["normalised_mse"],
                 sp.get("zq_mean"), up.get("zq_mean"), up.get("zq_mean_w"), up.get("hist"),
                 up.get("z_mean"), up.get("z_mean_w")))
# sort by best of (zq, zq_whitened) descending so the winner floats to the top
def _best(row): return max([v for v in (row[3], row[4]) if v is not None] or [0])
rows.sort(key=_best, reverse=True)
nan=float('nan')
print(f"{'config':32s} {'normMSE':>8} {'seenF1':>7} {'unseen(zq)':>11} {'unseen(zq_w)':>13} {'unseen(hist)':>13} {'unseen(zmean)':>14} {'unseen(zmean_w)':>16}")
for name,nm,s,uz,uzw,uh,um,umw in rows:
    uzw = uzw if uzw is not None else nan
    um  = um  if um  is not None else nan
    umw = umw if umw is not None else nan
    print(f"{name:32s} {nm:8.3f} {s:7.3f} {uz:11.3f} {uzw:13.3f} {uh:13.3f} {um:14.3f} {umw:16.3f}")
best=rows[0] if rows else None
if best and best[6] is not None:
    tok=max([v for v in (best[3],best[4]) if v is not None]); cont=max([v for v in (best[6],best[7]) if v is not None])
    gap=cont-tok
    print(f"\nBOTTLENECK READ (top config): continuous {cont:.3f} vs quantized {tok:.3f} (gap {gap:+.3f})")
    print("  gap > +0.03 => QUANTIZER loses unseen info (try RVQ / finer codes)."
          if gap > 0.03 else
          "  gap <= +0.03 => ENCODER is the ceiling (invest in SSL / pretrained-encoder distillation).")
PY
