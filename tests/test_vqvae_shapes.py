"""Forward-pass sanity on the activity-aware VQ-VAE."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from har_icl.phase1_vqvae.model import ActivityAwareVQVAE


def test_forward_pass_shapes_no_conditioning():
    model = ActivityAwareVQVAE(
        num_codes=128, num_aux_classes=10,
        use_conditioning=False, use_aux_classifier=True,
    )
    x = torch.randn(4, 6, 128)
    lbl = torch.zeros(4, dtype=torch.long)
    out = model(x, label=lbl)
    assert out["x_hat"].shape == x.shape
    assert out["indices"].shape == (4, 16)
    for k in ("recon_loss", "commit_loss", "aux_loss", "perplexity", "loss"):
        assert out[k].dim() == 0, f"{k} should be scalar, got {tuple(out[k].shape)}"
    assert float(out["perplexity"]) >= 1.0
    assert float(out["perplexity"]) <= 128.0 + 1e-3


def test_forward_pass_with_conditioning():
    model = ActivityAwareVQVAE(
        num_codes=64, num_aux_classes=5,
        use_conditioning=True, text_dim=32, use_aux_classifier=True,
    )
    x = torch.randn(2, 6, 128)
    text = torch.randn(2, 32)
    lbl = torch.zeros(2, dtype=torch.long)
    out = model(x, text_emb=text, label=lbl)
    assert out["x_hat"].shape == x.shape
    assert out["indices"].shape == (2, 16)


def test_backward_pass_updates_parameters():
    model = ActivityAwareVQVAE(
        num_codes=32, num_aux_classes=4,
        use_conditioning=False, use_aux_classifier=True,
    )
    x = torch.randn(3, 6, 128)
    lbl = torch.tensor([0, 1, 2], dtype=torch.long)
    out = model(x, label=lbl)
    out["loss"].backward()
    # At least one trainable parameter has a gradient.
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    assert has_grad
