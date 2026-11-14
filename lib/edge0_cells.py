"""Preflight-only cell for Edge0 latent-MoE models.

These models are NOT llama.cpp/GGUF: they need Edge0's Strata engine built
from source (CUDA), a ~92-105 GB asset pack, and >= 24 GB host RAM. This cell
checks the Kaggle host and explains the verdict instead of pretending to run.
"""

CELL_EDGE0_PREFLIGHT = r'''
# --- Cell 2: Edge0 preflight (no llama.cpp here) ---------------------------
import os, shutil, subprocess

print("model:", "__PROJECT_NAME__")
print("pack :", "__MODEL_REPO__")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
free = shutil.disk_usage("/kaggle/temp").free / 1e9
nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
print(gpu or "no gpu")
print(f"RAM {ram:.0f} GB | /kaggle/temp free {free:.0f} GB | nvcc {nvcc}")

print("""
Edge0 needs:
  * the Strata engine built from github.com/Edge0-AI/Edge0 (CUDA 13 recipe;
    sm_75 is supported, but the build takes ~20-40 min here)
  * a 92-105 GB pack (use the -q3k repo, not -NVFP4: T4 has no FP4 hardware)
  * >= 16 GiB VRAM for one GPU + ~20 GiB RAM for the expert cache
Expected on 2x T4 (estimate): 5-15 tok/s decode, 80-250 tok/s prefill.
This kernel only reports the verdict; see edge0-*/README.md for the manual
recipe and expected numbers.
""", flush=True)
print("PREFLIGHT_DONE", flush=True)
'''
