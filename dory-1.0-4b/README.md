# NVIDIA Dory 1.0 4B — Remote GPU Inference on Kaggle

Runs **NVIDIA Nemotron Labs Dory 1.0 4B** (quantized to Q5_K_M, ~3.42 GB) with `llama.cpp` on a free Kaggle GPU instance, exposing an OpenAI-compatible API and Web UI via Cloudflare tunnel.

## Model Details
- **Architecture**: Looped Latent-Recurrent Architecture (4B effective parameter capacity)
- **Quantization**: Q5_K_M (~3.42 GB)
- **Source**: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF`
- **Context Length**: 32,768 tokens (tested up to 256K)
- **Inference Engine**: `llama.cpp` build `b11636`

## Quick Start

```bash
# Push and start the remote Kaggle kernel
./ctr.sh push

# Check live logs and wait for Cloudflare URL
./ctr.sh logs -f

# Check server health
./ctr.sh health

# Stop remote kernel when finished
./ctr.sh stop
```

## Endpoints
- `GET /` — llama.cpp Web UI
- `POST /v1/chat/completions` — OpenAI compatible Chat Completions API
- `GET /health` — Cluster health check
