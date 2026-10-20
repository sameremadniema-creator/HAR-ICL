"""sentence-BERT text encoder cache + FiLM conditioning module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """FiLM affine modulation: z_e <- z_e * (1 + gamma(t)) + beta(t)."""

    def __init__(self, text_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(text_dim, latent_dim)
        self.beta = nn.Linear(text_dim, latent_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, z_e: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Modulate (B, D, T) by (B, text_dim)."""
        g = self.gamma(text_emb).unsqueeze(-1)  # (B, D, 1)
        b = self.beta(text_emb).unsqueeze(-1)
        return z_e * (1.0 + g) + b


class TextEmbeddingCache:
    """Pre-compute and cache sentence-BERT embeddings per unified label."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_path = Path(cache_path) if cache_path else None
        self._embeddings: dict[str, np.ndarray] = {}
        self._dim: int | None = None
        self._model = None  # Lazy-loaded.
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Default for all-MiniLM-L6-v2.
            return 384
        return self._dim

    def _load_cache(self) -> None:
        data = np.load(self.cache_path, allow_pickle=True)
        keys = list(data["keys"])
        embs = data["embeddings"]
        self._embeddings = {k: embs[i] for i, k in enumerate(keys)}
        if embs.size:
            self._dim = embs.shape[1]

    def save(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._embeddings.keys())
        embs = np.stack([self._embeddings[k] for k in keys], axis=0) if keys else np.empty((0, self.dim))
        np.savez(self.cache_path, keys=np.array(keys, dtype=object), embeddings=embs)

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers not installed. `pip install sentence-transformers`."
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Return (N, D)."""
        texts = list(texts)
        missing = [t for t in texts if t not in self._embeddings]
        if missing:
            model = self._load_model()
            new_embs = model.encode(
                missing, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False
            )
            for t, e in zip(missing, new_embs):
                self._embeddings[t] = e.astype(np.float32)
            self._dim = new_embs.shape[1]
            self.save()
        return np.stack([self._embeddings[t] for t in texts], axis=0).astype(np.float32)

    def encode_dict(self, label_to_text: dict[str, str]) -> dict[str, np.ndarray]:
        """Encode each unique text once, return {label: embedding}."""
        texts = list(label_to_text.values())
        embs = self.encode(texts)
        return {lbl: embs[i] for i, lbl in enumerate(label_to_text.keys())}


def build_label_to_text(label_map: dict) -> dict[str, str]:
    """{unified_label: rich description}."""
    out: dict[str, str] = {}
    for lbl, entry in label_map.items():
        if lbl.startswith("_"):
            continue
        out[lbl] = entry.get("description", lbl) if isinstance(entry, dict) else lbl
    return out


def fallback_random_embedding(label_set, dim: int = 384, seed: int = 0) -> dict[str, np.ndarray]:
    """When sentence-transformers is unavailable (CI, offline) use a deterministic"""
    out: dict[str, np.ndarray] = {}
    for lbl in label_set:
        rng = np.random.default_rng(abs(hash((seed, lbl))) % (2**31))
        v = rng.normal(0, 1.0, size=dim).astype(np.float32)
        v /= max(np.linalg.norm(v), 1e-8)
        out[lbl] = v
    return out
