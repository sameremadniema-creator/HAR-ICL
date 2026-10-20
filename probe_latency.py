# probe_latency.py — R2.1: inference latency + peak GPU memory of the discrete.
import json, os, time, statistics as st
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from functools import partial

from phase3_gpt2.lora_setup import build_gpt2_with_lora
from phase3_gpt2.few_shot_dataset import FewShotICLDataset, collate_with_padding

R = Path(os.environ.get("HAR_ROOT", "/mnt/fast/nobackup/users/aa08742"))
TOKENS = R / "tokens/vqvae_usc_t32_s42"
CKPT   = R / "checkpoints/gpt2/vqvae_usc_t32_splitC_k4_r16_s42_matched/best"
K_SHOT, MAX_DISTR, CTX, N_EP, WARM = 4, 6, 1024, 200, 20

device = torch.device("cuda")
meta = json.loads((TOKENS / "meta.json").read_text())

base = build_gpt2_with_lora(num_codes=int(meta["num_codes"]),
                            activity_labels=list(meta["label_set"]),
                            use_lora=False)
base.model.resize_token_embeddings(len(base.tokenizer))
from peft import PeftModel
model = PeftModel.from_pretrained(base.model, str(CKPT)).to(device).eval()
bundle = base; bundle.model = model

ds = FewShotICLDataset(tokens_dir=TOKENS, bundle=bundle, split="C", k_shot=K_SHOT,
                       mode="test", max_distractors=MAX_DISTR, max_length=CTX,
                       seed=123, label_randomization=True)
loader = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=partial(collate_with_padding,
                                       pad_token_id=int(bundle.tokenizer.pad_token_id)))

torch.cuda.reset_peak_memory_stats()
lat, toks = [], []
with torch.no_grad():
    for i, b in enumerate(loader):
        if i >= N_EP + WARM: break
        ids = b["input_ids"].to(device); am = b["attention_mask"].to(device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _ = model(input_ids=ids, attention_mask=am)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if i >= WARM:
            lat.append((t1 - t0) * 1000.0); toks.append(int(am.sum().item()))

lat_s = sorted(lat)
out = {
  "gpu": torch.cuda.get_device_name(0),
  "episodes_timed": len(lat),
  "k_shot": K_SHOT,
  "tokens_per_episode_mean": round(st.mean(toks), 1),
  "latency_ms_mean": round(st.mean(lat), 2),
  "latency_ms_p50":  round(lat_s[len(lat)//2], 2),
  "latency_ms_p95":  round(lat_s[int(len(lat)*0.95)], 2),
  "peak_cuda_memory_MB": round(torch.cuda.max_memory_allocated()/2**20, 1),
  "fp32_weights_MB": round(sum(p.numel() for p in model.parameters())*4/2**20, 1),
  "fp16_weights_MB_est": round(sum(p.numel() for p in model.parameters())*2/2**20, 1),
}
print(json.dumps(out, indent=2))
Path("probe_latency_result.json").write_text(json.dumps(out, indent=2))
print("saved -> probe_latency_result.json")
