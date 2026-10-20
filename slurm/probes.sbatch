#!/bin/bash
#SBATCH --job-name=har_probes
#SBATCH --partition=3090
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/probes_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/aa08742/har_icl/logs/slurm/probes_%j.err

set -euo pipefail
ROOT=${HAR_ROOT:-/mnt/fast/nobackup/users/aa08742}
source ~/.bashrc
conda activate "$ROOT/envs/har_icl" || { echo "env missing"; exit 1; }
cd "$ROOT/har_icl"
nvidia-smi --query-gpu=name,power.limit --format=csv

echo "=====================  LATENCY PROBE  ====================="
python probe_latency.py

echo "=====================  ATTENTION PROBE ====================="
python probe_attention.py

echo "DONE. Results: probe_latency_result.json, probe_attention_result.json, supp_fig_s11_attention.png"
