# Baidu Unlimited-OCR-2 — Remote GPU Inference on Kaggle

Runs **Baidu Unlimited-OCR-2** on a Kaggle GPU instance. Unlimited-OCR-2 pioneers the OCR-3.0 era with Sparse Reference Sliding Window Attention (SR-SWA), achieving unbounded document parsing across single and multi-page documents, tables, charts, and financial forms without quadratic memory scaling.

## Features
- **Unlimited Horizon Parsing**: SR-SWA attention allows processing multi-page documents seamlessly.
- **Web Studio**: Modern interactive web interface with multi-page drag & drop, thumbnails preview, and copyable Markdown output.
- **REST & OpenAI Endpoints**:
  - `GET /` — Interactive Web Studio
  - `POST /v1/parse` — High-performance document parser endpoint (supports single or multiple images)
  - `POST /v1/chat/completions` — OpenAI compatible multimodal chat endpoint
  - `GET /health` — Health and GPU status check
- **Remote Control**: Cloudflare control tunnel enables PC control via `./ctr.sh stop`.

## Quick Start

```bash
# Push and start the remote Kaggle kernel
./ctr.sh push

# Stream logs and get the public Cloudflare tunnel URL
./ctr.sh logs -f

# Check health
./ctr.sh health

# Stop remote kernel when done
./ctr.sh stop
```

## API Usage

### Parse Endpoint (`/v1/parse`)
```bash
curl -X POST "https://<tunnel-url>/v1/parse" \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["data:image/png;base64,..."],
    "prompt": "Multi page parsing.",
    "max_new_tokens": 4096
  }'
```
