#!/bin/bash
set -euo pipefail

ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
ENV_PREFIX=${HAR_ENV:-$ROOT/envs/har_icl}
# Repo dir = parent of this script's directory.
REPO_DIR=${HAR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

echo "[setup] ROOT       = $ROOT"
echo "[setup] ENV_PREFIX = $ENV_PREFIX"
echo "[setup] REPO_DIR   = $REPO_DIR"

# 1.
echo "[setup] creating directories..."
mkdir -p "$REPO_DIR"/logs/slurm \
         "$ROOT"/data/raw "$ROOT"/data/processed \
         "$ROOT"/checkpoints/vqvae "$ROOT"/checkpoints/gpt2 "$ROOT"/checkpoints/baselines \
         "$ROOT"/tokens \
         "$REPO_DIR"/manifests

# 2.
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
           "$HOME/anaconda3/etc/profile.d/conda.sh" \
           "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$c" ]; then source "$c"; break; fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  # fall back to the cluster module system
  module load miniconda 2>/dev/null || module load anaconda3 2>/dev/null || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not found. Load or install conda (e.g. 'module avail' to find it)," >&2
  echo "       then re-run this script." >&2
  exit 1
fi

# 3.
if [ -d "$ENV_PREFIX" ]; then
  echo "[setup] env already exists at $ENV_PREFIX — skipping create"
else
  echo "[setup] creating conda env (a few minutes)..."
  conda env create -f "$REPO_DIR/environment.yaml" -p "$ENV_PREFIX"
fi

# 4.
source activate "$ENV_PREFIX" 2>/dev/null || conda activate "$ENV_PREFIX"

# 5.
echo "[setup] installing PyTorch (CUDA 12.1 build) ..."
pip install --no-cache-dir --force-reinstall \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 6.
echo "[setup] installing remaining Python deps ..."
pip install --no-cache-dir \
    "transformers==4.43.0" "peft==0.12.0" "accelerate==0.33.0" \
    "sentence-transformers==3.0.1" einops "hydra-core==1.3.2" omegaconf \
    wandb tensorboard ucimlrepo pytest pytest-xdist
# Chronos is a heavy OPTIONAL dep (one baseline).
pip install --no-cache-dir --no-deps "chronos-forecasting==1.5.1" \
    || echo "[setup] chronos-forecasting optional — skipped (baseline will use fallback)"
# Guarantee the tested core versions regardless of what optional deps tried to.
pip install --no-cache-dir "transformers==4.43.0" "tokenizers==0.19.1"

# 7.
pip install --no-cache-dir -e "$REPO_DIR"

# 8.
python - <<'PY'
import sys, torch
cu = torch.version.cuda
print(f"[setup] torch {torch.__version__} built for CUDA {cu}")
if not (cu and cu.startswith("12")):
    print(f"[setup] FATAL: torch is a CUDA {cu} build; AISurrey driver 560 needs CUDA 12.x.", file=sys.stderr)
    print("[setup] Re-run: pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 "
          "torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
    sys.exit(1)
print("[setup] OK: CUDA 12.x build — compatible with AISurrey driver 560.35.03 (CUDA <= 12.6).")
PY

echo
echo "[setup] DONE."
echo "[setup] Activate the env with:"
echo "          conda activate $ENV_PREFIX"
echo "[setup] Then smoke-test:"
echo "          cd $REPO_DIR && make test   # uses temp-dir fixtures; does NOT touch data/processed"
echo "[setup] (GPU is verified inside each job by scripts/assert_cuda.py — jobs FAIL"
echo "         loudly instead of silently running on CPU.)"
