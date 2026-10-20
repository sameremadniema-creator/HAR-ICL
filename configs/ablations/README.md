# Ablation matrix

`phase4_eval/ablation_runner.py` consumes every YAML in this directory and submits
one SLURM job per row. Numbering follows PDF Part 3.

| # | Config | Description |
|---|--------|-------------|
| 1 | `ablation_01_vanilla.yaml`  | VQ-VAE only — no conditioning, no aux loss |
| 2 | `ablation_02_aux_only.yaml` | + aux classification loss |
| 3 | `ablation_03_cond_only.yaml`| + sentence-BERT conditioning |
| 4 | `ablation_04_full.yaml`     | Full method |
| 5 | `ablation_05_codebook_sweep.yaml` | Codebook size 128/256/512/1024 |
| 6 | `ablation_06_lora_vs_ft.yaml`     | LoRA r=8 vs full GPT-2 fine-tuning |
| 7 | `ablation_07_k_sweep.yaml`        | K ∈ {1,2,4,8} |
| 8 | `ablation_08_continuous_proj.yaml` | Continuous projection baseline (headline ablation) |
| 9 | `ablation_09_zero_vs_few.yaml`    | K=0 vs few-shot |
|10 | `ablation_10_frozen_vs_trainable.yaml` | Frozen vs trainable codebook in Phase 3 |
