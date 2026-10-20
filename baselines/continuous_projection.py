"""Continuous-projection baseline = ablation #8 (SensorLLM-style)."""
from __future__ import annotations

import argparse
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from har_icl.phase1_vqvae.model import Encoder1D
from har_icl.phase3_gpt2.lora_setup import build_gpt2_with_lora, trainable_parameter_count
from har_icl.utils.metrics import macro_f1, per_class_f1, overfit_report
from har_icl.utils.seeds import set_global_seed

from ._common import load_split_arrays


class ReprogrammingHead(nn.Module):
    """Time-LLM-style reprogramming (Jin et al., ICLR 2024, arXiv:2310.01728)."""
    def __init__(self, latent_dim, llm_embed, llm_wte, n_proto=1000, n_heads=8, d_keys=None, dropout=0.1):
        super().__init__()
        # Exact match to the official Time-LLM ReprogrammingLayer + source-embedding.
        d_keys = d_keys or (latent_dim // n_heads)
        self.n_heads = n_heads
        vocab = llm_wte.weight.shape[0]
        self.mapping_layer = nn.Linear(vocab, n_proto)
        self.register_buffer("wte", llm_wte.weight.detach(), persistent=False)  # frozen vocab matrix; not in state_dict
        self.query_projection = nn.Linear(latent_dim, d_keys * n_heads)
        self.key_projection = nn.Linear(llm_embed, d_keys * n_heads)
        self.value_projection = nn.Linear(llm_embed, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, llm_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        proto = self.mapping_layer(self.wte.permute(1, 0)).permute(1, 0)        # (n_proto, llm_embed)
        B, L, _ = z.shape
        S, H = proto.shape[0], self.n_heads
        Q = self.query_projection(z).view(B, L, H, -1)
        K = self.key_projection(proto).view(S, H, -1)
        V = self.value_projection(proto).view(S, H, -1)
        scale = 1.0 / (Q.shape[-1] ** 0.5)
        scores = torch.einsum("blhe,she->bhls", Q, K)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", A, V).reshape(B, L, -1)
        return self.out_projection(out)


class ContinuousProjector(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm_wte: nn.Module,
        latent_dim: int = 64,
        gpt2_embed_dim: int = 768,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        # MATCHED control: `encoder` is the SAME pre-trained VQ-VAE encoder the discrete.
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = ReprogrammingHead(latent_dim, gpt2_embed_dim, llm_wte)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, T_lat, gpt2_embed_dim)."""
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)
        z = z.permute(0, 2, 1)               # (B, T_lat, D)
        return self.head(z)


class ContinuousICLDataset(Dataset):
    """Like FewShotICLDataset but emits raw window arrays alongside text token IDs."""

    def __init__(
        self,
        split_data: dict,
        bundle,
        k_shot: int,
        mode: str,
        max_distractors: int = 6,
        seed: int = 0,
        label_randomization: bool = True,
    ) -> None:
        self.k_shot = k_shot
        self.max_distractors = max_distractors
        self.bundle = bundle
        self.mode = mode
        # Matched to the discrete path: randomise class->answer-token per episode.
        self.label_randomization = bool(label_randomization)
        self._answer_token_pool = list(bundle.answer_token_ids.values())
        train = split_data.get("train", {})
        support_pool = split_data.get("support_pool", {})
        if mode == "train":
            self.query_X = train["X"]; self.query_y = train["y"]
        elif mode == "val":
            self.query_X = split_data["val"]["X"]; self.query_y = split_data["val"]["y"]
        else:
            self.query_X = split_data["test"]["X"]; self.query_y = split_data["test"]["y"]
        self.train_X = train.get("X", np.empty((0,)))
        self.train_y = train.get("y", np.empty((0,), dtype=object))
        self.support_X = np.concatenate([self.train_X, support_pool.get("X", np.empty((0, *self.train_X.shape[1:])))], axis=0) if support_pool else self.train_X
        self.support_y = np.concatenate([self.train_y, support_pool.get("y", np.empty((0,), dtype=object))], axis=0) if support_pool else self.train_y
        self.seen_labels = sorted(set(self.train_y.tolist()))
        self.unseen_labels = sorted(set(self.support_y.tolist()) - set(self.seen_labels))
        self.label_to_supports: dict[str, np.ndarray] = {}
        for lbl in set(self.support_y.tolist()):
            self.label_to_supports[lbl] = np.where(self.support_y == lbl)[0]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.query_y)

    def _candidate_labels(self, gold: str) -> list[str]:
        others = [l for l in self.seen_labels if l != gold]
        self.rng.shuffle(others)
        picks = others[: self.max_distractors]
        if gold not in picks:
            picks.insert(0, gold)
        self.rng.shuffle(picks)
        return picks

    def __getitem__(self, i):
        gold = str(self.query_y[i])
        q_x = torch.from_numpy(self.query_X[i].astype(np.float32))
        candidates = self._candidate_labels(gold)
        # Exclude the query from its own support: in train mode query_X is train["X"], and
        eqid = i if self.mode == "train" else None
        support_xs, support_lbls = [], []
        for lbl in candidates:
            pool = self.label_to_supports.get(lbl, np.array([], dtype=np.int64))
            if len(pool) == 0:
                continue
            pool_list = pool.tolist()
            if eqid is not None:
                pool_list = [p for p in pool_list if p != eqid]
                if not pool_list:
                    continue
            for _ in range(self.k_shot):
                pick = int(self.rng.choice(pool_list))
                support_xs.append(torch.from_numpy(self.support_X[pick].astype(np.float32)))
                support_lbls.append(lbl)
        # Per-episode class->answer-token map (meta-ICL), matched to the discrete path.
        classes = sorted({gold} | set(support_lbls))
        if self.label_randomization:
            toks = self.rng.sample(self._answer_token_pool, len(classes))
            label_to_token = {c: int(t) for c, t in zip(classes, toks)}
        else:
            label_to_token = {c: int(self.bundle.answer_token_ids[c]) for c in classes}
        tok2lbl = {int(t): c for c, t in label_to_token.items()}
        return {"query_x": q_x, "support_xs": support_xs, "support_lbls": support_lbls,
                "gold": gold, "label_to_token": label_to_token, "tok2lbl": tok2lbl,
                "cand_token_ids": sorted(tok2lbl.keys())}


def _collate(batch):
    return batch


def _build_inputs_embeds(
    example: dict,
    projector: ContinuousProjector,
    bundle,
    device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (inputs_embeds (1, L, D), labels (1, L), answer_position)."""
    text_emb_layer = bundle.model.get_input_embeddings()
    sp = bundle.special_token_ids
    ans_ids = bundle.answer_token_ids
    l2t = example.get("label_to_token")

    def ans_tok(lbl: str) -> int:
        return int(l2t[lbl]) if l2t is not None else int(ans_ids[lbl])

    def tok_emb(token_id: int) -> torch.Tensor:
        return text_emb_layer(torch.tensor([token_id], device=device)).squeeze(0)

    # Encode supports + query through projector in a single batch for efficiency.
    sup_x_list = example["support_xs"]
    n_sup = len(sup_x_list)
    if n_sup:
        sup_batch = torch.stack([x.transpose(0, 1) for x in sup_x_list], dim=0).to(device)
    q_batch = example["query_x"].transpose(0, 1).unsqueeze(0).to(device)
    if n_sup:
        all_emb = projector(torch.cat([sup_batch, q_batch], dim=0))
        sup_emb = all_emb[:n_sup]
        q_emb = all_emb[-1]
    else:
        q_emb = projector(q_batch)[0]
        sup_emb = None

    seq_emb: list[torch.Tensor] = [tok_emb(sp["<BOS_SIG>"])]
    label_ids: list[int] = [-100]
    for i, lbl in enumerate(example["support_lbls"]):
        seq_emb.append(tok_emb(sp["<ACT>"])); label_ids.append(-100)
        seq_emb.append(tok_emb(ans_tok(lbl))); label_ids.append(-100)
        for t in range(sup_emb.shape[1]):
            seq_emb.append(sup_emb[i, t]); label_ids.append(-100)
        seq_emb.append(tok_emb(sp["<SEP>"])); label_ids.append(-100)
    seq_emb.append(tok_emb(sp["<QUERY>"])); label_ids.append(-100)
    for t in range(q_emb.shape[0]):
        seq_emb.append(q_emb[t]); label_ids.append(-100)
    seq_emb.append(tok_emb(sp["<ANS>"])); label_ids.append(-100)
    answer_pos = len(seq_emb)
    gold_tok = ans_tok(example["gold"])
    seq_emb.append(tok_emb(gold_tok)); label_ids.append(gold_tok)

    inputs = torch.stack(seq_emb, dim=0).unsqueeze(0)
    labels = torch.tensor(label_ids, dtype=torch.long, device=device).unsqueeze(0)
    return inputs, labels, answer_pos


def run(args) -> dict:
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_split_arrays(Path(args.data), args.split)
    label_set = data["label_set"]

    # Build GPT-2 + LoRA but DO NOT use the sig_X tokens — continuous tokens flow
    bundle = build_gpt2_with_lora(
        num_codes=1,
        activity_labels=label_set,
        rank=args.lora_rank, alpha=2 * args.lora_rank, dropout=0.05,  # match discrete path (alpha=2*rank)
        target_modules=("c_attn",),
    )
    bundle.model.to(device)
    train_p, total_p = trainable_parameter_count(bundle.model)
    print(f"[continuous_projection] trainable {train_p:,} / {total_p:,}")
    gpt2_embed_dim = bundle.model.get_input_embeddings().weight.shape[1]
    in_ch = int(data["train"]["X"].shape[-1])
    vq_ckpt = getattr(args, "vqvae_ckpt", None)
    if vq_ckpt:
        # MATCHED control: reuse the SAME frozen, pre-trained VQ-VAE encoder as the.
        from har_icl.phase2_tokenize.encode_signals import _build_model_from_ckpt
        ck = torch.load(vq_ckpt, map_location=device)
        n_cls = len(ck.get("label_set", label_set))
        vqvae = _build_model_from_ckpt(ck, num_classes=n_cls).to(device).eval()
        latent_dim = int(ck["cfg"]["encoder"]["latent_dim"])
        projector = ContinuousProjector(vqvae.encoder, bundle.model.get_input_embeddings(),
                                        latent_dim=latent_dim, gpt2_embed_dim=gpt2_embed_dim,
                                        freeze_encoder=True).to(device)
        print(f"[continuous_projection] MATCHED frozen VQ-VAE encoder from {vq_ckpt} (latent={latent_dim}); "
              f"Time-LLM reprogramming bridge")
    else:
        # SMOKE-ONLY fallback: a fresh trainable encoder (NOT a matched control — never.
        from har_icl.phase1_vqvae.model import Encoder1D as _Enc
        print("[continuous_projection] WARNING: no --vqvae_ckpt; using a FRESH encoder (smoke only, NOT matched).")
        projector = ContinuousProjector(_Enc(in_ch, (64, 128, 256), 64, kernel_size=5),
                                        bundle.model.get_input_embeddings(),
                                        latent_dim=64, gpt2_embed_dim=gpt2_embed_dim, freeze_encoder=False).to(device)

    optim_params = [p for p in projector.parameters() if p.requires_grad] + \
                   [p for p in bundle.model.parameters() if p.requires_grad]
    # weight_decay defaults to 0.0 to MATCH the discrete path.
    optimiser = optim.AdamW(optim_params, lr=args.lr,
                            weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0))
    # grad_accum: accumulate this many episodes before an optimiser step, so the.
    accum = max(1, int(getattr(args, "grad_accum", 1) or 1))

    lr_flag = bool(getattr(args, "label_randomization", True))
    train_ds = ContinuousICLDataset(data, bundle, args.k_shot, mode="train", seed=args.seed, label_randomization=lr_flag)
    val_ds = ContinuousICLDataset(data, bundle, args.k_shot, mode="val", seed=args.seed, label_randomization=lr_flag)
    test_ds = ContinuousICLDataset(data, bundle, args.k_shot, mode="test", seed=args.seed, label_randomization=lr_flag)

    def _epoch(ds, train_mode: bool, cap: int = 0, ep: int = 0) -> dict:
        bundle.model.train(train_mode)
        projector.train(train_mode)
        order = list(range(len(ds)))
        if train_mode:
            # Per-epoch reshuffle (seed varies with epoch) so the model sees a DIFFERENT
            random.Random(args.seed + ep).shuffle(order)
        if cap and cap > 0:
            order = order[:cap]
        tot_loss, n = 0.0, 0
        preds, golds = [], []
        if train_mode:
            optimiser.zero_grad(set_to_none=True)
        for step, i in enumerate(order):
            ex = ds[i]
            inp, lab, ans_pos = _build_inputs_embeds(ex, projector, bundle, device)
            out = bundle.model(inputs_embeds=inp, labels=lab)
            if train_mode:
                # Gradient accumulation to reach effective batch = accum (matches discrete's 32).
                (out.loss / accum).backward()
                if (step + 1) % accum == 0 or (step + 1) == len(order):
                    optimiser.step(); optimiser.zero_grad(set_to_none=True)
            tot_loss += float(out.loss); n += 1
            if not train_mode:
                logits = out.logits[0, ans_pos - 1]
                # Per-episode decode: restrict to this episode's candidate answer tokens.
                cand = ex["cand_token_ids"]
                t2l = ex["tok2lbl"]
                mask = torch.full_like(logits, float("-inf"))
                for tid in cand:
                    mask[tid] = 0.0
                pred_id = int(torch.argmax(logits + mask).item())
                preds.append(t2l.get(pred_id, "<unk>")); golds.append(ex["gold"])
        return {"loss": tot_loss / max(1, n), "preds": preds, "golds": golds}

    # Best-checkpoint selection (match the discrete path, which selects.
    import copy as _copy
    trainable_keys = [k for k, p in bundle.model.named_parameters() if p.requires_grad]
    def _snapshot():
        sd = bundle.model.state_dict()
        return {"head": _copy.deepcopy(projector.head.state_dict()),
                "model": {k: sd[k].detach().cpu().clone() for k in trainable_keys}}
    def _restore(s):
        projector.head.load_state_dict(s["head"])
        sd = bundle.model.state_dict()
        for k, v in s["model"].items():
            sd[k].copy_(v.to(device))

    history = []
    best_blended = -1.0
    best_state = None
    best_ep = -1
    patience = int(getattr(args, "patience", 15))
    seen = set(train_ds.seen_labels)
    for ep in range(args.epochs):
        tr = _epoch(train_ds, True, cap=int(getattr(args, "max_train_episodes", 0) or 0), ep=ep)
        va = _epoch(val_ds, False, cap=int(getattr(args, "max_eval_episodes", 0) or 0))
        v_seen = [(g, p) for g, p in zip(va["golds"], va["preds"]) if g in seen]
        v_unseen = [(g, p) for g, p in zip(va["golds"], va["preds"]) if g not in seen]
        # same metric as the discrete eval: macro-F1 restricted to the gold label set
        m_seen = macro_f1([g for g,_ in v_seen], [p for _,p in v_seen], labels=sorted({g for g,_ in v_seen})) if v_seen else 0.0
        m_unseen = macro_f1([g for g,_ in v_unseen], [p for _,p in v_unseen], labels=sorted({g for g,_ in v_unseen})) if v_unseen else 0.0
        blended = 0.5 * (m_seen + m_unseen)
        record = {"epoch": ep, "train_loss": tr["loss"], "val_loss": va["loss"],
                  "val_seen_f1": m_seen, "val_unseen_f1": m_unseen, "val_blended_f1": blended}
        history.append(record)
        print(record, flush=True)
        if blended > best_blended:
            best_blended = blended
            best_state = _snapshot()
            best_ep = ep
        # early stop: the matched (frozen good-encoder) control converges well before.
        if patience > 0 and (ep - best_ep) >= patience:
            print(f"[early-stop] no val_blended improvement for {patience} epochs "
                  f"(best={best_blended:.4f} @ ep{best_ep}); stopping at ep{ep}.", flush=True)
            break

    overfit_report(history)

    if best_state is not None:
        _restore(best_state)

    te = _epoch(test_ds, False, cap=0)   # full test set for the reported metric
    t_seen = [(g, p) for g, p in zip(te["golds"], te["preds"]) if g in seen]
    t_unseen = [(g, p) for g, p in zip(te["golds"], te["preds"]) if g not in seen]
    labels = sorted(set(te["golds"]) | set(te["preds"]))
    result = {
        "model": "ContinuousProjection",
        "split": args.split, "seed": args.seed, "k_shot": args.k_shot,
        "best_val_blended_f1": float(best_blended),
        "overall_macro_f1": macro_f1(te["golds"], te["preds"], labels=labels),
        # same metric as the discrete eval (labels = gold set), so the comparison is.
        "seen_macro_f1": macro_f1([g for g,_ in t_seen], [p for _,p in t_seen], labels=sorted({g for g,_ in t_seen})) if t_seen else 0.0,
        "unseen_macro_f1": macro_f1([g for g,_ in t_unseen], [p for _,p in t_unseen], labels=sorted({g for g,_ in t_unseen})) if t_unseen else 0.0,
        "per_class_f1": per_class_f1(te["golds"], te["preds"], labels=labels),
        "trainable_params": int(train_p),
        "history": history,
    }
    out_path = Path(args.out or f"checkpoints/baselines/continuous_projection/split{args.split}_k{args.k_shot}_s{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/processed/v1")
    p.add_argument("--split", type=str, default="C")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vqvae_ckpt", type=str, default=None,
                   help="frozen pre-trained VQ-VAE checkpoint = the MATCHED encoder. Omit only for smoke tests.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=15,
                   help="early-stop after this many epochs without val_blended improvement (0 disables)")
    p.add_argument("--max_train_episodes", type=int, default=0,
                   help="cap training episodes per epoch (0 = all). Bounds epoch wall-time for the "
                        "one-episode-per-forward continuous control.")
    p.add_argument("--max_eval_episodes", type=int, default=0,
                   help="cap VALIDATION episodes per epoch (0 = all). Test set is always full.")
    p.add_argument("--k_shot", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=16)      # match the discrete path
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="AdamW weight decay; default 0.0 to match the discrete path.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="episodes accumulated per optimiser step = effective batch. Set to 32 to match "
                        "the discrete effective batch (8 x 4).")
    p.add_argument("--label_randomization", type=int, default=1,
                   help="1=meta-ICL randomised labels (match discrete path); 0=fixed labels")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    args.label_randomization = bool(args.label_randomization)
    run(args)


if __name__ == "__main__":
    main()
