# Qwen3.8-Flash-Next — Remote GPU Server on Kaggle

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

[← back to Remote GPU Servers](../README.md)

Runs Alibaba's **Qwen3.8-Flash-Next** (Qwen4 experimental foundation preview) multimodal MoE model with `llama.cpp` on Kaggle Dual Tesla T4 GPUs (30 GB usable VRAM), exposing an OpenAI-compatible API and Web UI via Cloudflare tunnel.

## Model Details
- **Developer**: Alibaba Cloud Qwen Team
- **Model**: `Qwen/Qwen3.8-Flash-Next` (125B MoE with ~6B active per token, 51B N-gram embedding table, 48 layers, 512 experts)
- **Architecture**: `qwen4_exp` / `Qwen4ExpForConditionalGeneration` (Hybrid Qwen Sparse Attention [QSA] + Gated DeltaNet [GDN] + Gated Residual [GR])
- **Quantization**: `UD-IQ4_XS` (~18.89 GB) with REAP25 (Redundant Expert Allocation Pruning) + BF16 mmproj (~0.85 GB)
- **Source**: `anik-jha/Qwen3.8-Flash-Next-24GB-GGUF`
- **Context Length**: 32,768 tokens (up to 262,144 tokens native)
- **Features**: Multimodal vision, micro-block sparse attention, deep `<think>` reasoning, agentic tool use
- **Inference Engine**: `llama.cpp` build `b12174` with multi-GPU layer pipelining across dual Tesla T4s

## Quantization Analysis for Kaggle GPUs

| Quantization Tier | Model Size | Kaggle Dual T4 Status | Expected Speed | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **`REAP25-UD-IQ4_XS`** | **18.89 GB** | **100% Fits in VRAM** (~9.5 GB / GPU) | **~25–40 tok/s** | ⭐ **Best / Recommended** |
| `UD-IQ1_S` | 61.82 GB | ⚠️ Needs 32 GB CPU RAM offload | ~0.8–1.5 tok/s | High perplexity loss |
| `UD-IQ2_XS` | 70.62 GB | ❌ OOM crash (exceeds 60GB total RAM+VRAM) | N/A | Crashes runner |
| `UD-IQ3_XXS` | 77.88 GB | ❌ Exceeds 73GB Kaggle disk limit | N/A | Download fails |
| `UD-Q4_K_XL` | 103.69 GB | ❌ Impossible on Kaggle | N/A | Needs 4x A100 |

## Quick Start

```bash
# Push and start the remote Kaggle kernel
./ctr.sh push

# Or push and wait for the Cloudflare URL:
../bin/cluster qwen3.8-flash-next up

# Launch OpenCode agent directly against the model:
../bin/cluster qwen3.8-flash-next agent

# Check server health
../bin/cluster qwen3.8-flash-next health

# Stop remote kernel when finished
../bin/cluster qwen3.8-flash-next stop
```

## Endpoints
- `GET /` — llama.cpp Web UI
- `POST /v1/chat/completions` — OpenAI compatible Chat Completions API
- `GET /health` — Cluster health check
- `GET /log` — Live server inference logs
