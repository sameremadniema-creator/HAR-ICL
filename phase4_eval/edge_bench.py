"""Edge-device latency / memory probe (PDF Framing A)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from har_icl.phase1_vqvae.model import ActivityAwareVQVAE
from har_icl.phase3_gpt2.lora_setup import build_gpt2_with_lora, trainable_parameter_count
from har_icl.phase3_gpt2.prompts import build_prompt
from har_icl.utils.checkpointing import load_torch_checkpoint


def _now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def bench_vqvae(ckpt_path: str, n_warmup: int = 5, n_iters: int = 50) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_torch_checkpoint(ckpt_path, map_location=str(device))
    cfg = ckpt["cfg"]
    label_set = ckpt.get("label_set", [])
    model = ActivityAwareVQVAE(
        in_channels=int(cfg["encoder"]["in_channels"]),
        hidden_channels=list(cfg["encoder"]["hidden_channels"]),
        latent_dim=int(cfg["encoder"]["latent_dim"]),
        num_codes=int(cfg["codebook"]["num_codes"]),
        codebook_decay=float(cfg["codebook"]["decay"]),
        commit_beta=float(cfg["loss"]["commit_beta"]),
        use_conditioning=bool(cfg["conditioning"]["enabled"]),
        text_dim=int(cfg["conditioning"]["text_dim"]),
        use_aux_classifier=bool(cfg["aux_classifier"]["enabled"]),
        num_aux_classes=len(label_set),
        encoder_strides=cfg["encoder"].get("strides"),
        quantizer_type=str(cfg["codebook"].get("type", "vq")),
        fsq_levels=cfg["codebook"].get("levels"),
        ssl_contrastive_weight=float(cfg["loss"].get("ssl_contrastive_weight", 0.0)),
        ssl_temperature=float(cfg["loss"].get("ssl_temperature", 0.1)),
        encoder_type=str(cfg["encoder"].get("type", "conv")),
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    x = torch.randn(1, int(cfg["encoder"]["in_channels"]), 128, device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        t = _now()
        for _ in range(n_iters):
            model(x)
        elapsed = _now() - t
    train_p, total_p = trainable_parameter_count(model)
    return {
        "component": "vqvae",
        "device": str(device),
        "params_trainable": int(train_p),
        "params_total": int(total_p),
        "median_latency_ms": float(elapsed / n_iters * 1000),
        "throughput_windows_per_s": float(n_iters / max(elapsed, 1e-9)),
        "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None,
    }


def bench_gpt2(ckpt_dir: str, num_codes: int, label_set, k_shot: int = 4, n_warmup: int = 3, n_iters: int = 20) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_gpt2_with_lora(
        num_codes=num_codes, activity_labels=list(label_set), use_lora=False,
    )
    bundle.model.resize_token_embeddings(len(bundle.tokenizer))
    try:
        from peft import PeftModel  # type: ignore

        bundle.model = PeftModel.from_pretrained(bundle.model, ckpt_dir)
    except Exception:
        pass
    bundle.model.to(device).eval()

    sig_per_window = 16
    support = []
    for j in range(k_shot):
        for lbl in list(label_set)[:3]:
            support.append((lbl, [j % num_codes] * sig_per_window))
    bp = build_prompt(bundle, support=support, query_indices=[0] * sig_per_window,
                      gold_label=list(label_set)[0], max_length=1024)
    input_ids = torch.tensor([bp.input_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(n_warmup):
            bundle.model(input_ids=input_ids, attention_mask=attn)
        t = _now()
        for _ in range(n_iters):
            bundle.model(input_ids=input_ids, attention_mask=attn)
        elapsed = _now() - t
    train_p, total_p = trainable_parameter_count(bundle.model)
    return {
        "component": "gpt2_lora",
        "device": str(device),
        "params_trainable": int(train_p),
        "params_total": int(total_p),
        "prompt_tokens": int(input_ids.size(1)),
        "median_latency_ms": float(elapsed / n_iters * 1000),
        "throughput_prompts_per_s": float(n_iters / max(elapsed, 1e-9)),
        "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--gpt2_ckpt", type=str, required=True)
    parser.add_argument("--tokens_meta", type=str, required=True,
                        help="Path to tokens/<run>/meta.json (for num_codes + label_set).")
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    meta = json.loads(Path(args.tokens_meta).read_text())
    label_set = list(meta["label_set"])
    num_codes = int(meta["num_codes"])

    results = {
        "vqvae": bench_vqvae(args.vqvae_ckpt),
        "gpt2": bench_gpt2(args.gpt2_ckpt, num_codes=num_codes, label_set=label_set, k_shot=args.k_shot),
    }
    out_path = Path(args.out) if args.out else Path(args.gpt2_ckpt).parent / "edge_bench.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
