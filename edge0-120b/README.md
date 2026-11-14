# Edge0-120B — EXPERIMENTAL (not llama.cpp)

Latent sparse MoE: 120B total / **6.1B active**, 48 layers, 512 experts,
256K ctx, native 4-bit. Runs on Edge0's **Strata** engine (built from source,
CUDA) with a ~92 GB pack. `./ctr.sh push` only runs a preflight.

## Kaggle T4 reality
- T4 = sm_75 -> no NVFP4; use `Edge0/Edge0-120B-q3k` (28 GB experts, 92 GB total).
- No prebuilt Linux engine -> compile it (~20-40 min on 4 vCPU).
- Needs 16 GiB VRAM + ~20 GB RAM -> one T4 + host RAM is enough.
- Kaggle image is CUDA 12.x; build with its nvcc for sm_75 (unproven).

## Expected speed on 2x T4 (estimate)
**5-15 tok/s decode, 80-250 tok/s prefill.** Edge0's own 16 GiB-VRAM capped
test on RTX 5090: 51.3 tok/s chat / 59.9 agent decode, 361 tok/s 16K prefill.
Upstream measurements (not T4): 5090D NVFP4 124.9/135.5, 4090 Q3_K 95.0/104.9,
M2 Ultra Q3_K 60.4/63.0.

## Manual recipe (supported host)
```bash
git clone https://github.com/Edge0-AI/Edge0.git
# build edge0-strata (edge0-strata/EDGE0.md) + edge0-infer (cpu preset)
hf download Edge0/Edge0-120B-q3k --local-dir pack
edge0-infer serve --model pack --device cuda:0 --context 32768
```
