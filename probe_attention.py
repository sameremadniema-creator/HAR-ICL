# probe_attention.py — R2.3: where does the reader look when classifying an.
import json, os
from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase3_gpt2.lora_setup import build_gpt2_with_lora
from phase3_gpt2.few_shot_dataset import FewShotICLDataset, collate_with_padding

R = Path(os.environ.get("HAR_ROOT", "/mnt/fast/nobackup/users/aa08742"))
TOKENS = R / "tokens/vqvae_usc_t32_s42"
CKPT   = R / "checkpoints/gpt2/vqvae_usc_t32_splitC_k4_r16_s42_matched/best"
K_SHOT, MAX_DISTR, CTX = 4, 6, 1024
PER_CLASS, MAX_SCAN, LAST_LAYERS = 6, 5000, 4   # stratified: 6 episodes per unseen class

device = torch.device("cuda")
meta = json.loads((TOKENS / "meta.json").read_text())

base = build_gpt2_with_lora(num_codes=int(meta["num_codes"]),
                            activity_labels=list(meta["label_set"]),
                            use_lora=False)
base.model.resize_token_embeddings(len(base.tokenizer))
from peft import PeftModel
model = PeftModel.from_pretrained(base.model, str(CKPT)).to(device).eval()
bundle = base; bundle.model = model
sp = bundle.special_token_ids
ACT, SEP, ANS = sp["<ACT>"], sp["<SEP>"], sp["<ANS>"]

ds = FewShotICLDataset(tokens_dir=TOKENS, bundle=bundle, split="C", k_shot=K_SHOT,
                       mode="test", max_distractors=MAX_DISTR, max_length=CTX,
                       seed=123, label_randomization=True)
unseen = set(ds.unseen_labels)
print(f"unseen labels: {sorted(unseen)}")
gen = torch.Generator(); gen.manual_seed(123)
loader = DataLoader(ds, batch_size=1, shuffle=True, generator=gen,
                    collate_fn=partial(collate_with_padding,
                                       pad_token_id=int(bundle.tokenizer.pad_token_id)))
quota = {c: PER_CLASS for c in unseen}

def first(x):
    return x[0] if isinstance(x, (list, tuple)) else x

records, example = [], None
with torch.no_grad():
    for i, b in enumerate(loader):
        if i >= MAX_SCAN or not any(quota.values()): break
        gold = first(b["gold_labels"])
        if gold not in unseen or quota.get(gold, 0) <= 0: continue
        quota[gold] -= 1
        ids = b["input_ids"].to(device); am = b["attention_mask"].to(device)
        seq = ids[0].tolist()
        n = int(am[0].sum().item())
        ans_idx = max(j for j in range(n) if seq[j] == ANS)      # the <ANS> token
        gold_tok = seq[ans_idx + 1]                              # episode's gold answer token
        tok2lbl = first(b["tok2lbl"]) or {}
        tok2lbl = {int(k): v for k, v in tok2lbl.items()} if isinstance(tok2lbl, dict) else {}

        out = model(input_ids=ids, attention_mask=am, output_attentions=True)
        # attention row of the <ANS> position, mean over heads, mean of last LAST_LAYERS.
        rows = [a[0, :, ans_idx, :].mean(0) for a in out.attentions[-LAST_LAYERS:]]
        w = torch.stack(rows).mean(0).float().cpu().numpy()      # (T,)
        cand = [int(c) for c in first(b["cand_token_ids"])]      # restrict to candidate answer tokens
        logit_row = out.logits[0, ans_idx]
        pred_tok = cand[int(torch.argmax(logit_row[cand]).item())] if cand else int(logit_row.argmax().item())

        # segment support pairs: <ACT> lbl sig*32 <SEP>
        acts = [j for j in range(ans_idx) if seq[j] == ACT]
        pairs = []
        for a_j in acts:
            s_j = next(j for j in range(a_j + 1, ans_idx) if seq[j] == SEP)
            ptok = seq[a_j + 1]
            pairs.append({"span": (a_j, s_j), "tok": ptok,
                          "label": tok2lbl.get(ptok, str(ptok)),
                          "is_gold_class": ptok == gold_tok,
                          "mass": float(w[a_j:s_j + 1].sum()),
                          "tok_w": w[a_j:s_j + 1].tolist()})
        tot = sum(p["mass"] for p in pairs) or 1.0
        for p in pairs: p["frac"] = p["mass"] / tot
        gold_frac = sum(p["frac"] for p in pairs if p["is_gold_class"])
        n_gold = sum(p["is_gold_class"] for p in pairs)
        rec = {"episode": i, "gold": gold,
               "pred": tok2lbl.get(pred_tok, str(pred_tok)),
               "correct": pred_tok == gold_tok,
               "n_pairs": len(pairs), "n_gold_pairs": n_gold,
               "uniform_share": n_gold / len(pairs),
               "gold_class_attention_frac": gold_frac,
               "pairs": [{k: p[k] for k in ("label", "is_gold_class", "frac")} for p in pairs]}
        records.append(rec)
        if example is None and rec["correct"]:
            example = {"rec": rec, "pairs": pairs}
        print(f"[{len(records):02d}] gold={gold:<16} pred={rec['pred']:<16} "
              f"correct={rec['correct']} gold-attn={gold_frac:.3f} (uniform {rec['uniform_share']:.3f})")

g = [r["gold_class_attention_frac"] for r in records]
u = [r["uniform_share"] for r in records]
summary = {"episodes": len(records),
           "mean_gold_class_attention_frac": round(float(np.mean(g)), 4),
           "sd": round(float(np.std(g)), 4),
           "mean_uniform_share": round(float(np.mean(u)), 4),
           "accuracy_in_probe": round(float(np.mean([r["correct"] for r in records])), 4),
           "mean_gold_frac_when_correct":  round(float(np.mean([r["gold_class_attention_frac"] for r in records if r["correct"]]) if any(r["correct"] for r in records) else 0), 4),
           "mean_gold_frac_when_wrong":    round(float(np.mean([r["gold_class_attention_frac"] for r in records if not r["correct"]]) if any(not r["correct"] for r in records) else 0), 4),
           "per_class": {c: {"n": len(rs),
                             "acc": round(float(np.mean([r["correct"] for r in rs])), 3),
                             "gold_attn": round(float(np.mean([r["gold_class_attention_frac"] for r in rs])), 3),
                             "preds": {p: sum(1 for r in rs if r["pred"] == p) for p in sorted({r["pred"] for r in rs})}}
                         for c in sorted(unseen)
                         for rs in [[r for r in records if r["gold"] == c]] if rs}}
print(json.dumps(summary, indent=2))
Path("probe_attention_result.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2))

# Supplementary Fig.
fig = plt.figure(figsize=(12, 9))
# (a) aggregate: gold-class vs other-class attention share across episodes
ax = fig.add_subplot(2, 2, 1)
other = [1 - x for x in g]
ax.bar([0, 1], [np.mean(g), np.mean(other)],
       yerr=[np.std(g), np.std(other)], capsize=4, color=["#d62728", "#7f7f7f"])
ax.axhline(np.mean(u), ls="--", c="k", lw=1, label=f"uniform share ({np.mean(u):.2f})")
ax.set_xticks([0, 1]); ax.set_xticklabels(["same-class\nsupport pairs", "other-class\nsupport pairs"])
ax.set_ylabel("fraction of support attention"); ax.legend(frameon=False)
ax.set_title(f"(a) Attention at <ANS>, {len(records)} unseen-gold episodes")

if example is not None:
    pr, rec = example["pairs"], example["rec"]
    # (b) per-pair attention in one correct unseen episode
    ax = fig.add_subplot(2, 2, 2)
    cols = ["#d62728" if p["is_gold_class"] else "#7f7f7f" for p in pr]
    ax.bar(range(len(pr)), [p["frac"] for p in pr], color=cols)
    ax.axhline(1/len(pr), ls="--", c="k", lw=1)
    ax.set_xlabel("support pair (prompt order)"); ax.set_ylabel("attention fraction")
    ax.set_title(f"(b) Example episode — gold: {rec['gold']} (red = same class)")
    # (c) token-level heatmap within each support pair
    ax = fig.add_subplot(2, 1, 2)
    M = np.array([p["tok_w"] for p in pr])
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xlabel("token within support pair (<ACT>, label, 32 sensor tokens, <SEP>)")
    ax.set_ylabel("support pair")
    ax.set_yticks(range(len(pr)))
    ax.set_yticklabels([("* " if p["is_gold_class"] else "") + p["label"] for p in pr], fontsize=6)
    fig.colorbar(im, ax=ax, label="attention weight")
    ax.set_title("(c) Token-level attention from <ANS> (same episode)")

fig.suptitle("Supplementary Figure S11 — attention of the in-context reader on unseen activities", y=0.995)
fig.tight_layout()
fig.savefig("supp_fig_s11_attention.png", dpi=300, bbox_inches="tight")
print("saved -> probe_attention_result.json, supp_fig_s11_attention.png")
