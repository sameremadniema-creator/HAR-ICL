"""Loss composition: recon (MSE + 0.1*L1) + commit + aux CE."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mse_weight: float = 1.0,
    l1_weight: float = 0.1,
) -> torch.Tensor:
    """Combined MSE + L1 reconstruction loss (PDF Part 2.6: 'MSE alone over-smooths"""
    return mse_weight * F.mse_loss(x_hat, x) + l1_weight * F.l1_loss(x_hat, x)


def aux_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al."""
    device = features.device
    f = F.normalize(features, dim=1)
    sim = f @ f.t() / temperature                      # (B, B)
    # Numerical stability.
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()).float().to(device)
    self_mask = torch.eye(f.size(0), device=device)
    pos_mask = pos_mask - self_mask                    # exclude self
    exp_sim = torch.exp(sim) * (1.0 - self_mask)       # exclude self from denom
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(dim=1)
    # Only average over anchors that have >=1 positive.
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.zeros((), device=device)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
    return -mean_log_prob_pos.mean()


def protonet_episodic_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    k_shot: int = 4,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Episodic prototypical-network loss (Snell et al."""
    device = features.device
    protos, q_feats, q_targets = [], [], []
    classes = labels.unique()
    proto_idx = 0
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if idx.numel() < k_shot + 1:
            continue
        perm = idx[torch.randperm(idx.numel(), device=device)]
        support, query = perm[:k_shot], perm[k_shot:]
        protos.append(features[support].mean(dim=0))
        q_feats.append(features[query])
        q_targets.append(torch.full((query.numel(),), proto_idx, device=device, dtype=torch.long))
        proto_idx += 1
    if proto_idx < 2:
        return torch.zeros((), device=device)
    P = torch.stack(protos, dim=0)          # (C, D)
    Q = torch.cat(q_feats, dim=0)           # (Nq, D)
    t = torch.cat(q_targets, dim=0)         # (Nq,)
    logits = -torch.cdist(Q, P, p=2).pow(2) / temperature
    return F.cross_entropy(logits, t)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Label-free NT-Xent / SimCLR loss over two augmented views (Chen et al."""
    device = z1.device
    B = z1.size(0)
    if B < 2:
        return torch.zeros((), device=device)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)                      # (2B, D)
    sim = z @ z.t() / temperature                       # (2B, 2B)
    self_mask = torch.eye(2 * B, device=device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, float("-inf"))
    targets = (torch.arange(2 * B, device=device) + B) % (2 * B)  # i <-> i+B
    return F.cross_entropy(sim, targets)
