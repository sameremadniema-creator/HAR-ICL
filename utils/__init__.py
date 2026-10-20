"""Shared utilities: seeds, metrics, logging, checkpointing, manifest."""
from .seeds import set_global_seed  # noqa: F401
from .metrics import (  # noqa: F401
    macro_f1,
    per_class_f1,
    codebook_perplexity,
    token_entropy,
    bootstrap_ci,
)
