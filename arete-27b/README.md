# NousResearch Arete-27B — Remote GPU Inference on Kaggle

Runs **NousResearch/Arete-27B** (quantized to Q4_K_M, ~16.02 GB + 0.87 GB multimodal projector) with `llama.cpp` on a free Kaggle GPU instance, exposing an OpenAI-compatible API and Web UI via Cloudflare tunnel.

## Model Details
- **Developer**: Nous Research
- **Model**: Arete-27B (post-trained from Alibaba's Qwen3.8-27B)
- **Features**: SFT value-alignment for freedom of expression, uncensored unbiased answers, robust agentic tool use, and deep reasoning
- **Quantization**: Q4_K_M (~16.02 GB) + BF16 mmproj (~0.87 GB)
- **Source**: `bartowski/NousResearch_Arete-27B-GGUF`
- **Context Length**: 32,768 tokens (trained context 64k)
- **Inference Engine**: `llama.cpp` build `b12174` with multimodal image/vision support

## Quick Start

```bash
# Push and start the remote Kaggle kernel
./ctr.sh push

# Stream live logs and wait for Cloudflare URL
./ctr.sh logs -f

# Check server health
./ctr.sh health

# Stop remote kernel when finished
./ctr.sh stop
```

## Endpoints
- `GET /` — llama.cpp Web UI
- `POST /v1/chat/completions` — OpenAI compatible Chat Completions API (supports text, images, tool calling, and `<think>` reasoning)
- `GET /health` — Cluster health check
