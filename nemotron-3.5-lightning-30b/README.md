# NVIDIA-Nemotron-3.5-Lightning-30B-A3B — Remote GPU Server on Kaggle

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

[← back to Remote GPU Servers](../README.md)

Runs NVIDIA's **NVIDIA-Nemotron-3.5-Lightning-30B-A3B** hybrid Mamba-2 + MoE + attention language model (official `Q4_0` GGUF, ~18.90 GB) with `llama.cpp` on Kaggle Dual Tesla T4 GPUs (30 GB VRAM), exposing an OpenAI-compatible API and Web UI via Cloudflare tunnel.

## Model Details
- **Developer**: NVIDIA
- **Model**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (30B total parameters / ~3B active parameters per token)
- **Architecture**: Hybrid Mamba-2 + MoE + attention (52 layers: 26 Mamba-2, 20 MoE, 6 attention) with MTP heads
- **Quantization**: Official `Q4_0` GGUF (~18.90 GB) from the ggml-org conversion of the BF16 reference weights
- **Source**: `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`
- **Context Length**: 32,768 tokens served here (natively up to 256k–1M; only ~6 KB of KV cache per token, so raise `__CTX__` freely)
- **Features**: Reasoning mode on/off via chat template (`enable_thinking`), agentic tool calling, 6 languages, very low active-parameter cost
- **Inference Engine**: `llama.cpp` build `b12360` (`nemotron_h_moe` architecture support)

> This folder serves the **deployable GGUF** of the BF16 reference checkpoint. For
> NVIDIA's own optimized data-centre path, see the `NVFP4` release (needs Hopper /
> Blackwell — not the T4s Kaggle provides).

## Quick Start

```bash
# Push and start the remote Kaggle kernel
./ctr.sh push

# Or push and wait for the Cloudflare URL:
../bin/cluster nemotron-3.5-lightning-30b up

# Launch OpenCode agent directly against the model:
../bin/cluster nemotron-3.5-lightning-30b agent

# Check server health
../bin/cluster nemotron-3.5-lightning-30b health

# Measure real prefill/decode tok/s (server-side timings)
../bin/cluster nemotron-3.5-lightning-30b bench --long 16000

# Stop remote kernel when finished
../bin/cluster nemotron-3.5-lightning-30b stop
```

## Endpoints
- `GET /` — llama.cpp Web UI
- `POST /v1/chat/completions` — OpenAI compatible Chat Completions API (supports tool calling and `enable_thinking` reasoning toggles)
- `GET /health` — Cluster health check

## Tuning notes
- **Speculative decoding (the big one)**: this GGUF repo ships
  `mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (~1.16 GB). llama.cpp
  auto-detects it as `draft-mtp` from the GGUF metadata, so it is a one-liner:

  ```bash
  __DRAFT_FILE__="mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf" \
  __DRAFT_SIZE_GB__=1.16 ./ctr.sh push
  ```

  Expect ~1.3–2× decode. Run `./ctr.sh bench` afterwards and check the reported
  draft acceptance rate — below ~40% it is not worth the extra VRAM.
- **Measure, don't guess**: `./ctr.sh bench --long 16000` reports the server's own
  prompt/decode tok/s and greps the log for CPU-offloaded layers.
- **Smaller / larger quants**: swap `__MODEL_FILE__` (e.g. `...-Q8_0.gguf` is 33.59 GB and
  does **not** fit dual T4; `bartowski/...-IQ4_XS.gguf` is ~18.92 GB).
- **Prefill**: `-ub 512` is conservative; try `-ub 1024` if `bench` shows free VRAM.
- **Flash attention** stays `off` on sm_75 to avoid the T4 prefill deadlock seen on
  other projects; re-enable with `__FLASH_ATTN__=on ./ctr.sh push` if you want to test.
