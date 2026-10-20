# Re-running SensorLLM-3b on our splits

Per PDF Part 4 we re-run SensorLLM's Stage-2 Task-Aware Tuning against the same
held-out-by-name splits. Budget: ~4 GPU-days. Their codebase is heavy so we
document the procedure rather than vendoring it.

```bash
# 1. Clone SensorLLM and follow their setup (LLaMA-3.2-3B + Chronos-base).
git clone https://github.com/cruiseresearchgroup/SensorLLM
cd SensorLLM
# follow their environment.yaml — needs LLaMA weights via HF.

# 2. Replace their dataset config with our harmonised splits.
#    Their dataloader expects a JSONL of {window, label} pairs.
python scripts/export_for_sensorllm.py \
  --data ../data/processed/v1 \
  --split C \
  --out ../data/processed/v1/sensorllm_jsonl

# 3. Run their Stage-2 fine-tune on the SEEN classes only (their classifier head
#    is fixed; it cannot predict unseen classes).
bash scripts/run_stage2.sh --data ../data/processed/v1/sensorllm_jsonl --split C

# 4. For our held-out-by-name benchmark we report:
#    - Seen-class macro F1: their classifier output as normal.
#    - Unseen-class macro F1: 0.0 by construction (PDF Framing C) — we cite
#      SensorLLM's Limitations section to make this concrete.
```

The eval CSV from their run goes into `manifests/sensorllm_3b_split{X}_s{seed}.csv`
and is merged into Table 1 by `phase4_eval.ablation_runner --mode collect`.
