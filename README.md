# Discrete Sensor Tokens for In-Context Recognition of Unseen Human Activities

Code and result files accompanying the manuscript *"Discrete Sensor Tokens for In-Context Recognition of Unseen Human Activities"* (Al-Ibadi et al., under review at *Scientific Reports*).

## Overview

A class-discriminative VQ-VAE converts each 2.56 s IMU window (6 channels, 50 Hz, 128 samples) into 32 discrete codes from a 512-entry codebook. A frozen GPT-2 (124M) with LoRA adapters (rank 16) reads labelled support examples and a query as discrete tokens and answers by copying the label of the matching support example. The class-to-answer-token mapping is randomized in every episode, so the protocol is vocabulary-free and the model cannot rely on class names.

A matched continuous control (Time-LLM reprogramming over the same frozen encoder, identical optimization protocol) varies only the discrete-versus-continuous interface.

## Main results (device-constant USC-HAD split, 8 seeds, mean ± population s.d.)

| Result | Value | Source in this repository |
|---|---|---|
| Discrete model, unseen macro-F1 (K = 4) | 0.478 ± 0.046 | `results/per_seed/discrete/discrete_k4_s*_eval.json`, `results/slurm_logs/matched_2167352_*.out` |
| Discrete model, unseen accuracy / seen macro-F1 | 0.319 ± 0.042 / 0.941 | same files |
| Matched continuous control, unseen macro-F1 | 0.278 ± 0.042 (discrete wins 8/8 seeds, Wilcoxon p = 0.0078) | `results/per_seed/continuous/`, `matched_2167352_*.out` |
| Shot scaling K = 1 / 2 / 4 / 8 | 0.480 / 0.508 / 0.478 / 0.505 | `discrete_k{1,2,4,8}_s*_eval.json`, `kshotm_2167353_*.out` |
| All-unseen episodes (macro-F1 / accuracy) | 0.465 / 0.464 | `discrete_k4_s*_eval_all_unseen.json`, `evalunseen_2167354_*.out` |
| Prototypical network / few-shot fine-tuning | 0.347 ± 0.088 / 0.168 ± 0.038 | `base_2168688_*.out` |
| Continuous control learning-rate sweep | see paper | `clr_2168692_*.out` |
| Continuous control with fixed labels (seen macro-F1) | 0.955 | `m1_full_2199494.out` |
| Tokenizer design sweep and device-confound analysis | see paper (Table 4, Section 5.1) | `results/sweep_confound/` |
| Inference cost (RTX 3090, FP32, batch 1) | 20.8 ms per episode, 1.41 GB peak memory | `results/probes/probe_latency_result.json` |
| Attention on same-class support pairs | 0.33 overall; 0.67 correct vs 0.13 wrong | `results/probes/probe_attention_result.json` |

Empirical chance level: 0.249 macro-F1 / 0.142 accuracy.

## Repository layout

```
configs/          YAML configs (data, splits, vqvae, gpt2, ablations)
phase0_data/      dataset download, harmonization to 50 Hz windows, subject-wise splits
phase1_vqvae/     class-discriminative VQ-VAE tokenizer (EMA codebook, aux CE, SupCon)
phase2_tokenize/  frozen tokenizer -> discrete code sequences
phase3_gpt2/      GPT-2 + LoRA in-context reader, prompt construction, episode sampling
phase4_eval/      evaluation (macro-F1, accuracy, chance level, all-unseen episodes)
baselines/        matched continuous control, ProtoNet, fixed-head and fine-tuning baselines
scripts/          dataset download, result harvesting, cluster helpers
slurm/            SLURM job scripts used for the paper (RTX 3090 partition)
tests/            unit tests (label mapping, leakage guards, prompt budget)
utils/            seeds, metrics, logging, checkpointing
probe_latency.py, probe_attention.py   inference-cost and attention analyses (Sections 6.1 and 6.3)
results/          per-seed result files, SLURM logs, probe outputs and figures
```

## Reproducing the results

1. Create the environment from `environment_lock.yaml` (or `requirements.txt`), then run `pip install -e .`.
2. Download the public datasets: `python scripts/download_datasets.py --raw-dir data/raw` (USC-HAD may need a manual download from https://sipi.usc.edu/had/).
3. Run the unit tests: `pytest tests/ -q`.
4. Submit the SLURM jobs. The scripts read the root paths from the environment variables `HAR_ROOT`, `HAR_REPO` and `HAR_ENV`; set them for your cluster.
   - `slurm/matched_run.sbatch`: discrete model and matched continuous control, 8 seeds
   - `slurm/kshot_matched.sbatch`: shot-scaling curve
   - `slurm/eval_all_unseen.sbatch`: all-unseen episodes
   - `slurm/baselines_rerun.sbatch`: baselines
   - `slurm/continuous_lr_sweep.sbatch`: continuous control learning-rate sweep
   - `slurm/codebook_sweep.sbatch`, `slurm/confound.sbatch`: tokenizer sweep and confound analysis
   - `slurm/probes.sbatch`: inference-cost and attention probes

Seeds: 42, 1337, 2024, 7, 31337, 99, 2025, 31415.

## Datasets

The raw datasets are not included in this repository. They are available from their original sources: UCI-HAR (https://archive.ics.uci.edu/dataset/240), PAMAP2 (https://archive.ics.uci.edu/dataset/231), MHealth (https://archive.ics.uci.edu/dataset/319) and USC-HAD (https://sipi.usc.edu/had/).
