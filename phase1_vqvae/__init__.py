"""Phase 1 — Activity-aware VQ-VAE."""
from .codebook import VectorQuantizerEMA  # noqa: F401
from .conditioning import FiLMConditioning, TextEmbeddingCache  # noqa: F401
from .model import (  # noqa: F401
    ActivityAwareVQVAE,
    Encoder1D,
    Decoder1D,
    AuxClassifier,
)
