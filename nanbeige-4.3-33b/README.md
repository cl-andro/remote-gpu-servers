# Nanbeige 4.3 33B — Remote GPU Inference on Kaggle

Runs **Nanbeige4.3-33B** (quantized to Q4_K_M, ~19.36 GB) with `llama.cpp` on a free Kaggle GPU instance, exposing an OpenAI-compatible API and Web UI via Cloudflare tunnel.

## Model Details
- **Architecture**: Causal Language Model (33B parameters)
- **Quantization**: Q4_K_M (~19.36 GB)
- **Source**: `bartowski/Nanbeige_Nanbeige4.3-33B-GGUF`
- **Context Length**: 32,768 tokens
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
