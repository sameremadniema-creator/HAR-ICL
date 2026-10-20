"""EMA-updated vector quantiser with k-means warm-up."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes: int,
        dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.beta = commitment_beta

        embed = torch.randn(num_codes, dim) * 0.01
        self.register_buffer("embedding", embed)            # (K, D)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed_avg", embed.clone())
        self.register_buffer("initialised", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def kmeans_init(self, z_e_flat: torch.Tensor) -> None:
        """Initialise codebook from k-means on encoder outputs."""
        N, D = z_e_flat.shape
        assert D == self.dim, f"dim mismatch: {D} vs {self.dim}"
        # Sample initial centroids.
        if N >= self.num_codes:
            idx = torch.randperm(N, device=z_e_flat.device)[: self.num_codes]
        else:
            # Pad by replicating.
            idx = torch.arange(self.num_codes, device=z_e_flat.device) % N
        centroids = z_e_flat[idx].clone()
        for _ in range(10):
            d = torch.cdist(z_e_flat, centroids)        # (N, K)
            assign = d.argmin(dim=1)                    # (N,)
            for k in range(self.num_codes):
                mask = assign == k
                if mask.any():
                    centroids[k] = z_e_flat[mask].mean(dim=0)
        self.embedding.copy_(centroids)
        self.ema_embed_avg.copy_(centroids)
        self.ema_cluster_size.fill_(1.0)
        self.initialised.fill_(True)

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise (B, D, T) → (z_q (B,D,T), indices (B,T), commit_loss, perplexity)."""
        B, D, T = z_e.shape
        assert D == self.dim, f"channel dim {D} != codebook dim {self.dim}"
        flat = z_e.permute(0, 2, 1).contiguous().reshape(-1, D)  # (B*T, D)

        if not bool(self.initialised.item()) and self.training:
            self.kmeans_init(flat.detach())

        # Squared L2 distance to each code (K, D).
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )  # (B*T, K)
        indices_flat = dist.argmin(dim=1)
        indices = indices_flat.view(B, T)

        # Code counts via bincount (avoids materialising a (B*T, K) one-hot -> much lower.
        counts = torch.bincount(indices_flat, minlength=self.num_codes).type(flat.dtype)  # (K,)

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                embed_sum = torch.zeros_like(self.embedding)          # (K, D)
                embed_sum.index_add_(0, indices_flat, flat)           # == one_hot.t() @ flat
                self.ema_embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                # Laplace smoothing for cluster sizes.
                n = self.ema_cluster_size.sum()
                smoothed = ((self.ema_cluster_size + self.eps) / (n + self.num_codes * self.eps)) * n
                self.embedding.copy_(self.ema_embed_avg / smoothed.unsqueeze(1))

        z_q_flat = F.embedding(indices_flat, self.embedding)            # (B*T, D)
        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1).contiguous()      # (B, D, T)

        # Commitment loss only (EMA path skips codebook loss).
        commit_loss = self.beta * F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator.
        z_q = z_e + (z_q - z_e).detach()

        # Codebook perplexity for monitoring (exp of entropy in nats).
        probs = counts / counts.sum().clamp_min(1)
        nz = probs > 0
        entropy = -(probs[nz] * torch.log(probs[nz])).sum()
        perplexity = torch.exp(entropy)

        return z_q, indices, commit_loss, perplexity


class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization (Mentzer et al., ICLR 2024, arXiv:2309.15505)."""

    def __init__(self, dim: int, levels) -> None:
        super().__init__()
        levels = [int(v) for v in levels]
        assert all(v >= 2 for v in levels), "each FSQ level must be >= 2"
        self.dim = dim
        self.n_dim = len(levels)
        self.num_codes = int(math.prod(levels))
        # Per-dim level counts (long) and mixed-radix basis for index packing.
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long))
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.long), dim=0)
        self.register_buffer("_basis", basis)  # (n_dim,)
        # Project latent_dim <-> n_dim so encoder/decoder stay unchanged (they work in.
        self.in_proj = nn.Conv1d(dim, self.n_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(self.n_dim, dim, kernel_size=1)

    @staticmethod
    def _round_ste(z: torch.Tensor) -> torch.Tensor:
        """Round with a straight-through gradient (grad flows as identity)."""
        return z + (z.round() - z).detach()

    def _bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Squash R -> a bounded range so rounding lands in [-(L-1)/2, (L-1)/2]."""
        L = self._levels.float()
        half_l = (L - 1) * (1 + eps) / 2
        offset = torch.where(L % 2 == 0, torch.full_like(L, 0.5), torch.zeros_like(L))
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """(...,n_dim) -> quantized codes normalised to ~[-1,1], with STE."""
        q = self._round_ste(self._bound(z))
        half_width = (self._levels // 2).float()
        return q / half_width

    def _codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Normalised codes (...,n_dim) -> flat integer index in [0, num_codes)."""
        half_width = (self._levels // 2).float()
        shifted = codes * half_width + half_width            # back to [0, L-1]
        return (shifted.round().long() * self._basis).sum(dim=-1)

    def _perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_codes).float()
        probs = counts / counts.sum().clamp(min=1.0)
        nz = probs > 0
        entropy = -(probs[nz] * torch.log(probs[nz])).sum()
        return torch.exp(entropy)

    @torch.no_grad()
    def codebook(self) -> torch.Tensor:
        """Materialise the implicit codebook as (num_codes, dim)."""
        device = self.in_proj.weight.device
        idx = torch.arange(self.num_codes, device=device)                 # (K,)
        # Unpack mixed-radix index -> per-dim level integer in [0, L-1].
        per_dim = (idx.unsqueeze(1) // self._basis.unsqueeze(0)) % self._levels.unsqueeze(0)  # (K, n_dim)
        half_width = (self._levels // 2).float()
        codes = (per_dim.float() - half_width) / half_width               # (K, n_dim) normalised
        return self.out_proj(codes.unsqueeze(-1)).squeeze(-1)             # (K, dim)

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise (B, D, T) → (z_q (B,D,T), indices (B,T), commit_loss=0, perplexity)."""
        B, D, T = z_e.shape
        assert D == self.dim, f"channel dim {D} != FSQ dim {self.dim}"
        h = self.in_proj(z_e).transpose(1, 2)                 # (B, T, n_dim)
        codes = self.quantize(h)                              # (B, T, n_dim), STE
        indices = self._codes_to_indices(codes)              # (B, T)
        z_q = self.out_proj(codes.transpose(1, 2).contiguous())  # (B, D, T)
        # FSQ has no commitment loss (gradient flows through round_ste + projections).
        commit_loss = torch.zeros((), device=z_e.device, dtype=z_e.dtype)
        perplexity = self._perplexity(indices)
        return z_q, indices, commit_loss, perplexity
