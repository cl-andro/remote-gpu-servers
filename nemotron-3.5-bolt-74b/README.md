# NVIDIA-Nemotron-3.5-Bolt-74B-A9B — Remote GPU Server on Kaggle

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

[← back to Remote GPU Servers](../README.md)

Runs NVIDIA's **NVIDIA-Nemotron-3.5-Bolt-74B-A9B** hybrid Mamba2-Transformer MoE vision-language model (quantized 24GB footprint, ~20.51 GB GGUF + 1.61 GB multimodal projector) with `llama.cpp` on Kaggle Dual Tesla T4 GPUs (30 GB VRAM), exposing an OpenAI-compatible API and Web UI via Cloudflare tunnel.

## Model Details
- **Developer**: NVIDIA
- **Model**: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B` (74B total parameters / ~9B active parameters per token)
- **Architecture**: Hybrid Mamba2 + Transformer with LatentMoE and C-RADIO vision encoder
- **Quantization**: Native 24GB memory-optimized GGUF (~20.51 GB) + mmproj (~1.61 GB)
- **Source**: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF`
- **Context Length**: 32,768 tokens (up to 128k–256k native)
- **Features**: Multimodal vision (image, screen, video), deep reasoning & thinking budgets, agentic tool calling, high-throughput active routing
- **Inference Engine**: `llama.cpp` build `b12174` with multimodal image/vision support

## Quick Start

```bash
# Push and start the remote Kaggle kernel
./ctr.sh push

# Or push and wait for the Cloudflare URL:
../bin/cluster nemotron-3.5-bolt-74b up

# Launch OpenCode agent directly against the model:
../bin/cluster nemotron-3.5-bolt-74b agent

# Check server health
../bin/cluster nemotron-3.5-bolt-74b health

# Stop remote kernel when finished
../bin/cluster nemotron-3.5-bolt-74b stop
```

## Endpoints
- `GET /` — llama.cpp Web UI
- `POST /v1/chat/completions` — OpenAI compatible Chat Completions API (supports text, images, tool calling, and reasoning)
- `GET /health` — Cluster health check
