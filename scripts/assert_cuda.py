#!/usr/bin/env python3
"""assert_cuda.py — fail LOUDLY if CUDA isn't usable, so a SLURM job never silently"""
import sys

try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"[cuda-check] FATAL: cannot import torch: {e}", file=sys.stderr)
    sys.exit(2)

built_cu = torch.version.cuda
avail = torch.cuda.is_available()
print(f"[cuda-check] torch {torch.__version__} | built for CUDA {built_cu} | cuda_available={avail}")

if not avail:
    print("[cuda-check] FATAL: CUDA is NOT available — refusing to run on CPU.", file=sys.stderr)
    print("[cuda-check] On AISurrey this usually means a torch/driver mismatch.", file=sys.stderr)
    print("[cuda-check] Driver 560.35.03 supports CUDA <= 12.6; reinstall the cu121 build:", file=sys.stderr)
    print("[cuda-check]   pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 "
          "torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
    sys.exit(1)

if built_cu and not built_cu.startswith("12"):
    print(f"[cuda-check] FATAL: torch built for CUDA {built_cu}, but AISurrey needs CUDA 12.x "
          f"(driver 560 ⇒ CUDA <= 12.6).", file=sys.stderr)
    sys.exit(1)

print(f"[cuda-check] OK: using {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB)")
sys.exit(0)
