PY ?= python
SEED ?= 42
EPOCHS ?= 200
K ?= 4
SPLIT ?= C
RANK ?= 8
CODEBOOK ?= 512
EXP ?= vqvae_full_k$(CODEBOOK)
DATA_DIR ?= data/processed/v1
CKPT_DIR ?= checkpoints

.PHONY: help install data download download_status download_force test train_vqvae tokenize train_gpt2 eval ablations baselines lint clean

help:
	@echo "HAR-ICL Makefile targets:"
	@echo "  install            pip install -e . (with optional extras)"
	@echo "  download           fetch missing real datasets (idempotent — skips if present)"
	@echo "  download_status    report which datasets are already on disk; no network"
	@echo "  download_force     re-download even if already present"
	@echo "  data               download + harmonise real datasets to $(DATA_DIR)"
	@echo "  test               run pytest"
	@echo "  train_vqvae        EPOCHS=$(EPOCHS) SEED=$(SEED) EXP=$(EXP)"
	@echo "  tokenize           encode all splits with frozen VQ-VAE EXP=$(EXP) SEED=$(SEED)"
	@echo "  train_gpt2         SPLIT=$(SPLIT) K=$(K) RANK=$(RANK) SEED=$(SEED)"
	@echo "  eval               evaluate best GPT-2 ckpt on test split"
	@echo "  baselines          train all baselines"
	@echo "  ablations          run ablation matrix"

install:
	$(PY) -m pip install -e ".[torch,data,viz,logging,dev]"

# NOTE: synthetic data generation was REMOVED entirely (target + utils/synth.py).
# It used to overwrite the real data at $(DATA_DIR), silently corrupting every
# result. Use `make data` (real datasets) only.

# ── Dataset download (separate, idempotent — skips datasets already present) ──
RAW_DIR ?= data/raw

download:
	$(PY) scripts/download_datasets.py --raw-dir $(RAW_DIR)

download_status:
	$(PY) scripts/download_datasets.py --raw-dir $(RAW_DIR) --status

download_force:
	$(PY) scripts/download_datasets.py --raw-dir $(RAW_DIR) --force

data: download
	$(PY) -m phase0_data.load_and_harmonize \
		--config configs/data_v1.yaml \
		--raw-dir $(RAW_DIR) \
		--out $(DATA_DIR) \
		--skip-missing

test:
	$(PY) -m pytest tests/ -q

train_vqvae:
	$(PY) -m phase1_vqvae.train \
		--config configs/vqvae/$(EXP).yaml \
		--data $(DATA_DIR) \
		--seed $(SEED) \
		--epochs $(EPOCHS) \
		--ckpt_dir $(CKPT_DIR)/vqvae/$(EXP)_s$(SEED)

tokenize:
	$(PY) -m phase2_tokenize.encode_signals \
		--vqvae_ckpt $(CKPT_DIR)/vqvae/$(EXP)_s$(SEED)/best.pt \
		--data $(DATA_DIR) \
		--out tokens/$(EXP)_s$(SEED)

train_gpt2:
	$(PY) -m phase3_gpt2.train \
		--tokens tokens/$(EXP)_s$(SEED) \
		--split $(SPLIT) --k_shot $(K) --lora_rank $(RANK) --seed $(SEED) \
		--out $(CKPT_DIR)/gpt2/split$(SPLIT)_k$(K)_r$(RANK)_s$(SEED)

eval:
	$(PY) -m phase4_eval.eval \
		--ckpt $(CKPT_DIR)/gpt2/split$(SPLIT)_k$(K)_r$(RANK)_s$(SEED)/best \
		--tokens tokens/$(EXP)_s$(SEED) \
		--split $(SPLIT) --k_shot $(K)

baselines:
	$(PY) -m baselines.deepconvlstm --data $(DATA_DIR) --split $(SPLIT) --seed $(SEED)
	$(PY) -m baselines.deepconvlstm_attn --data $(DATA_DIR) --split $(SPLIT) --seed $(SEED)
	$(PY) -m baselines.attend --data $(DATA_DIR) --split $(SPLIT) --seed $(SEED)
	$(PY) -m baselines.protonet --data $(DATA_DIR) --split $(SPLIT) --seed $(SEED) --k_shot $(K)
	$(PY) -m baselines.continuous_projection --data $(DATA_DIR) --split $(SPLIT) --seed $(SEED) --k_shot $(K)

ablations:
	$(PY) -m phase4_eval.ablation_runner --configs configs/ablations --data $(DATA_DIR)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
