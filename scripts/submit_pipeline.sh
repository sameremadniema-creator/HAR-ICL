#!/bin/bash
set -euo pipefail

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

mkdir -p "$REPO_DIR"/logs/slurm        # must exist before sbatch parses --output

MODE=${1:-validate}
SPLIT=${SPLIT:-C}
K=${K:-4}

# Optionally exclude known-bad nodes (e.g.
EXCLUDE=${EXCLUDE:-}
EX_ARG=""
[ -n "$EXCLUDE" ] && EX_ARG="--exclude=$EXCLUDE" && echo "[submit] excluding nodes: $EXCLUDE"

echo "[submit] mode=$MODE split=$SPLIT k=$K root=$ROOT"

# Phase 0 — data (CPU / debug partition)
jid0=$(sbatch --parsable $EX_ARG slurm/phase0_data.sbatch)
echo "[submit] phase0 (data)      -> $jid0"

# Phase 1 — VQ-VAE, single seed 42 (the codebook is trained once)
jid1=$(SEED=42 sbatch --parsable $EX_ARG --dependency=afterok:$jid0 slurm/phase1_vqvae.sbatch)
echo "[submit] phase1 (vqvae s42) -> $jid1"

# Phase 2 — tokenise with that single VQ-VAE
jid2=$(SEED=42 sbatch --parsable $EX_ARG --dependency=afterok:$jid1 slurm/phase2_tokenize.sbatch)
echo "[submit] phase2 (tokenize)  -> $jid2"

if [ "$MODE" = "validate" ]; then
  # one GPT-2 seed (array index 0 = seed 42), then eval it
  jid3=$(SPLIT=$SPLIT K=$K sbatch --parsable $EX_ARG --dependency=afterok:$jid2 \
           --array=0 slurm/phase3_gpt2_array.sbatch)
  echo "[submit] phase3 (gpt2 s42)  -> $jid3"
  jid4=$(SPLIT=$SPLIT K=$K sbatch --parsable $EX_ARG --dependency=afterok:$jid3 \
           --array=0 slurm/phase4_eval.sbatch)
  echo "[submit] phase4 (eval s42)  -> $jid4"
else
  # all 5 GPT-2 seeds (array 0-4) — all read the single VQ_SEED=42 tokens
  jid3=$(SPLIT=$SPLIT K=$K sbatch --parsable $EX_ARG --dependency=afterok:$jid2 \
           slurm/phase3_gpt2_array.sbatch)
  echo "[submit] phase3 (gpt2 x5)   -> $jid3"
  jid4=$(SPLIT=$SPLIT K=$K sbatch --parsable $EX_ARG --dependency=afterok:$jid3 \
           slurm/phase4_eval.sbatch)
  echo "[submit] phase4 (eval x5)   -> $jid4"
fi

echo "[submit] queued. Monitor with:  squeue -u \$USER"
