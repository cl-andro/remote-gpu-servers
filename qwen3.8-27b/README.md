# Qwen3.8-27B — Remote GPU Server on Kaggle

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

[← back to Remote GPU Servers](../README.md)

Runs Alibaba's **Qwen3.8-27B** vision-language reasoning model (quantized to `UD-Q4_K_XL`, ~17.56 GB + 0.93 GB BF16 multimodal projector) with `llama.cpp` on Kaggle Dual Tesla T4 GPUs (30 GB VRAM), exposing an OpenAI-compatible API and Web UI via Cloudflare tunnel.

## Model Details
- **Developer**: Alibaba Cloud Qwen Team
- **Model**: `Qwen/Qwen3.8-27B` (dense vision-language model with hybrid linear-attention + full-attention)
- **Quantization**: `UD-Q4_K_XL` (~17.56 GB) + BF16 mmproj (~0.93 GB) by Unsloth
- **Source**: `unsloth/Qwen3.8-27B-GGUF`
- **Context Length**: 32,768 tokens (up to 262,144 tokens native)
- **Features**: Native vision (image and text understanding), deep reasoning `<think>` tags, agentic tool calling, developer role support
- **Inference Engine**: `llama.cpp` build `b12174` with multimodal image/vision support

## Quick Start

```bash
# Push and start the remote Kaggle kernel
./ctr.sh push

# Or push and wait for the Cloudflare URL:
../bin/cluster qwen3.8-27b up

# Launch OpenCode agent directly against the model:
../bin/cluster qwen3.8-27b agent

# Check server health
../bin/cluster qwen3.8-27b health

# Stop remote kernel when finished
../bin/cluster qwen3.8-27b stop
```

## Endpoints
- `GET /` — llama.cpp Web UI
- `POST /v1/chat/completions` — OpenAI compatible Chat Completions API (supports text, images, tool calling, and `<think>` reasoning)
- `GET /health` — Cluster health check
