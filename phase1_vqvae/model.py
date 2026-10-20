"""Activity-aware VQ-VAE."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import FSQQuantizer, VectorQuantizerEMA
from .conditioning import FiLMConditioning
from .losses import aux_ce_loss, info_nce_loss, protonet_episodic_loss, reconstruction_loss, supcon_loss


def _conv_block(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.Sequential:
    pad = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


def _deconv_block(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 2) -> nn.Sequential:
    pad = (kernel_size - 1) // 2
    out_pad = stride - 1
    return nn.Sequential(
        nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=out_pad
        ),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.GELU(),
    )


class Encoder1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels=(64, 128, 256),
        latent_dim: int = 64,
        kernel_size: int = 5,
        strides=None,
    ) -> None:
        super().__init__()
        if strides is None:
            strides = [2] * len(hidden_channels)
        chs = [in_channels, *hidden_channels]
        blocks = []
        for i in range(len(hidden_channels)):
            blocks.append(_conv_block(chs[i], chs[i + 1], kernel_size=kernel_size, stride=int(strides[i])))
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(hidden_channels[-1], latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        h = self.blocks(x)
        return self.proj(h)


class _SSMBlock(nn.Module):
    """Mamba-style selective state-space block (S6, Gu & Dao 2023) — pure PyTorch."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_kernel: int = 4) -> None:
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, conv_kernel, padding=conv_kernel - 1, groups=d_inner)
        self.dt_rank = max(1, d_model // 16)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        # softplus(bias) ~= 0.01 at init -> small, stable step sizes (Mamba's dt_init.
        nn.init.constant_(self.dt_proj.bias, -4.6)
        # S4D-real init: A = -diag(1..d_state) per channel, stored as log for positivity.
        self.A_log = nn.Parameter(torch.log(torch.arange(1.0, d_state + 1).repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def _scan(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, T, d_inner) -> selective scan with input-dependent (delta, B, C).
        Bsz, T, _ = u.shape
        dbc = self.x_proj(u)
        dt, Bp, Cp = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                      # (B, T, d_inner)
        A = -torch.exp(self.A_log)                                # (d_inner, d_state), Re(A)<0
        h = u.new_zeros(Bsz, self.d_inner, self.d_state)
        ys = []
        for t in range(T):
            d_t = delta[:, t].unsqueeze(-1)                       # (B, d_inner, 1)
            h = torch.exp(d_t * A) * h + d_t * Bp[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            ys.append((h * Cp[:, t].unsqueeze(1)).sum(-1))        # (B, d_inner)
        y = torch.stack(ys, dim=1)
        return y + self.D * u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        res = x
        x = self.norm(x)
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u)
        y = self._scan(u) + self._scan(u.flip(1)).flip(1)         # bidirectional
        y = y * F.silu(z)
        return res + self.out_proj(y)


class MambaEncoder1D(nn.Module):
    """Drop-in alternative to Encoder1D: same strided-conv stem (so tokens/window and"""

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels=(64, 128, 256),
        latent_dim: int = 64,
        kernel_size: int = 5,
        strides=None,
        ssm_layers: int = 2,
        ssm_state: int = 16,
    ) -> None:
        super().__init__()
        if strides is None:
            strides = [2] * len(hidden_channels)
        chs = [in_channels, *hidden_channels]
        blocks = []
        for i in range(len(hidden_channels)):
            blocks.append(_conv_block(chs[i], chs[i + 1], kernel_size=kernel_size, stride=int(strides[i])))
        self.blocks = nn.Sequential(*blocks)
        d_model = hidden_channels[-1]
        self.ssm = nn.ModuleList([_SSMBlock(d_model, d_state=ssm_state) for _ in range(ssm_layers)])
        self.proj = nn.Conv1d(d_model, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        h = self.blocks(x)                  # (B, d_model, T')
        h = h.transpose(1, 2)               # (B, T', d_model)
        for layer in self.ssm:
            h = layer(h)
        return self.proj(h.transpose(1, 2))  # (B, latent_dim, T')


class Decoder1D(nn.Module):
    def __init__(
        self,
        out_channels: int = 6,
        hidden_channels=(256, 128, 64),
        latent_dim: int = 64,
        kernel_size: int = 5,
        strides=None,
    ) -> None:
        super().__init__()
        if strides is None:
            strides = [2] * len(hidden_channels)
        self.proj = nn.Conv1d(latent_dim, hidden_channels[0], kernel_size=1)
        blocks = []
        for i in range(len(hidden_channels) - 1):
            blocks.append(
                _deconv_block(hidden_channels[i], hidden_channels[i + 1], kernel_size=kernel_size, stride=int(strides[i]))
            )
        blocks.append(
            _deconv_block(hidden_channels[-1], hidden_channels[-1], kernel_size=kernel_size, stride=int(strides[-1]))
        )
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv1d(hidden_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)
        h = self.blocks(h)
        return self.out(h)


class AuxClassifier(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # z_q: (B, D, T) → mean-pool over T → (B, D)
        pooled = z_q.mean(dim=-1)
        return self.net(pooled)


class ActivityAwareVQVAE(nn.Module):
    """Composes encoder + (optional) FiLM + VQ-EMA + decoder + (optional) aux head."""

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels=(64, 128, 256),
        latent_dim: int = 64,
        num_codes: int = 512,
        codebook_decay: float = 0.99,
        commit_beta: float = 0.25,
        use_conditioning: bool = True,
        text_dim: int = 384,
        use_aux_classifier: bool = True,
        num_aux_classes: int = 17,
        recon_l1_weight: float = 0.1,
        aux_ce_weight: float = 0.5,
        kernel_size: int = 5,
        encoder_strides=None,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        quantizer_type: str = "vq",
        fsq_levels=None,
        ssl_contrastive_weight: float = 0.0,
        ssl_temperature: float = 0.1,
        proto_weight: float = 0.0,
        proto_k_shot: int = 4,
        proto_temperature: float = 1.0,
        encoder_type: str = "conv",
    ) -> None:
        super().__init__()
        # ARCHITECTURAL KNOB — like strides/quantizer_type/ssl_proj, encoder_type MUST be
        self.encoder_type = str(encoder_type)
        if self.encoder_type == "mamba":
            self.encoder = MambaEncoder1D(in_channels, hidden_channels, latent_dim, kernel_size, strides=encoder_strides)
        else:
            self.encoder = Encoder1D(in_channels, hidden_channels, latent_dim, kernel_size, strides=encoder_strides)
        self.quantizer_type = quantizer_type
        if quantizer_type == "fsq":
            assert fsq_levels is not None, "quantizer_type='fsq' requires codebook.levels"
            self.quantizer = FSQQuantizer(dim=latent_dim, levels=fsq_levels)
            num_codes = self.quantizer.num_codes  # implicit codebook = prod(levels)
        else:
            self.quantizer = VectorQuantizerEMA(
                num_codes=num_codes,
                dim=latent_dim,
                decay=codebook_decay,
                commitment_beta=commit_beta,
            )
        dec_strides = list(reversed(encoder_strides)) if encoder_strides is not None else None
        self.decoder = Decoder1D(
            out_channels=in_channels,
            hidden_channels=tuple(reversed(hidden_channels)),
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            strides=dec_strides,
        )
        self.film = FiLMConditioning(text_dim, latent_dim) if use_conditioning else None
        self.aux = AuxClassifier(latent_dim, num_aux_classes) if use_aux_classifier else None
        self.recon_l1_weight = recon_l1_weight
        self.aux_ce_weight = aux_ce_weight
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.use_contrastive = float(contrastive_weight) > 0.0
        # Label-free SimCLR/NT-Xent head: projects mean-pooled encoder latent -> 128-d.
        self.ssl_weight = float(ssl_contrastive_weight)
        self.ssl_temperature = float(ssl_temperature)
        self.use_ssl = self.ssl_weight > 0.0
        self.ssl_proj = (
            nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, 128))
            if self.use_ssl else None
        )
        # Episodic ProtoNet loss (Snell'17) on the mean-pooled latent: trains the EXACT
        self.proto_weight = float(proto_weight)
        self.proto_k_shot = int(proto_k_shot)
        self.proto_temperature = float(proto_temperature)
        self.use_proto = self.proto_weight > 0.0
        self.use_conditioning = use_conditioning
        self.use_aux_classifier = use_aux_classifier
        self.num_codes = num_codes
        self.latent_dim = latent_dim

    def encode(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_q (B,D,T'), indices (B,T'))."""
        z_e = self.encoder(x)
        if self.film is not None and text_emb is not None:
            z_e = self.film(z_e, text_emb)
        z_q, indices, _, _ = self.quantizer(z_e)
        return z_q, indices

    def forward(
        self,
        x: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        x2: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Returns: dict(x_hat, indices, recon_loss, commit_loss, aux_loss, perplexity, loss)."""
        if x.dim() == 3 and x.shape[-1] != 1 and x.shape[1] != self.encoder.blocks[0][0].in_channels:
            # Common mistake: (B, T, C) instead of (B, C, T).
            if x.shape[2] == self.encoder.blocks[0][0].in_channels:
                x = x.transpose(1, 2).contiguous()
        z_e = self.encoder(x)
        if self.film is not None and text_emb is not None:
            z_e = self.film(z_e, text_emb)
        z_q, indices, commit_loss, perplexity = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        # Pad/crop to original length.
        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode="linear", align_corners=False)
        recon_loss = reconstruction_loss(x, x_hat, l1_weight=self.recon_l1_weight)
        # Supervised-loss row selection (see loss_mask in the docstring).
        sup = loss_mask.to(torch.bool) if loss_mask is not None else None
        sup_any = bool(sup.any()) if sup is not None else True
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.aux is not None and label is not None and sup_any:
            if sup is None:
                aux_loss = aux_ce_loss(self.aux(z_q), label)
            else:
                aux_loss = aux_ce_loss(self.aux(z_q[sup]), label[sup])
        contrastive_loss = torch.tensor(0.0, device=x.device)
        if self.use_contrastive and label is not None and sup_any:
            pooled = z_e.mean(dim=-1)  # (B, D) encoder-latent summary
            if sup is None:
                contrastive_loss = supcon_loss(pooled, label, self.contrastive_temperature)
            else:
                contrastive_loss = supcon_loss(pooled[sup], label[sup], self.contrastive_temperature)
        ssl_loss = torch.tensor(0.0, device=x.device)
        if self.use_ssl and x2 is not None:
            z_e2 = self.encoder(x2)                       # second augmented view
            h1 = self.ssl_proj(z_e.mean(dim=-1))
            h2 = self.ssl_proj(z_e2.mean(dim=-1))
            ssl_loss = info_nce_loss(h1, h2, self.ssl_temperature)
        proto_loss = torch.tensor(0.0, device=x.device)
        if self.use_proto and label is not None and sup_any:
            pooled = z_e.mean(dim=-1)
            p_feats = pooled if sup is None else pooled[sup]
            p_label = label if sup is None else label[sup]
            proto_loss = protonet_episodic_loss(
                p_feats, p_label, k_shot=self.proto_k_shot, temperature=self.proto_temperature)
        loss = (recon_loss + commit_loss + self.aux_ce_weight * aux_loss
                + self.contrastive_weight * contrastive_loss
                + self.ssl_weight * ssl_loss
                + self.proto_weight * proto_loss)
        return {
            "x_hat": x_hat,
            "indices": indices,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "aux_loss": aux_loss,
            "contrastive_loss": contrastive_loss,
            "ssl_loss": ssl_loss,
            "proto_loss": proto_loss,
            "perplexity": perplexity,
            "loss": loss,
        }
