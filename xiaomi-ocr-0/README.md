# Xiaomi-OCR-0 — Remote GPU Inference on Kaggle

Runs **Xiaomi-OCR-0** (by Xiaomi Research / SeerRay Lab) on a Kaggle GPU instance. Xiaomi-OCR-0 is a unified 0.8B OCR-specialized vision-language model trained on an ~170M sample corpus with Q-Mask text anchoring and Mix-RL. It achieved **#1 ranking** on Real5-OmniDocBench (95.24), **#2 ranking** on OmniDocBench v1.6 (96.83), and state-of-the-art results on formulas, tables, and document visual question answering.

## Features
- **Compact & High Performance**: 0.8B parameter vision-language model running at full BF16/FP16 precision on a single Kaggle GPU.
- **Multimodal Document Intelligence**:
  - Full document Markdown conversion with OTSL table parsing and LaTeX formulas in reading order.
  - Plain text extraction, table recognition, formula recognition, and structured JSON Key Information Extraction (KIE).
  - OCR-oriented Visual Question Answering (DocVQA, InfoVQA, ChartQA).
- **Interactive Web Studio**:
  - Drag & drop document image uploader (supports multiple pages).
  - Preset extraction modes (Full Markdown, Text, OTSL Tables, LaTeX, KIE JSON, Custom VQA).
  - One-click "Copy Output" button.
- **REST & OpenAI Endpoints**:
  - `GET /` — Interactive Web Studio
  - `POST /v1/parse` — High-speed document parsing endpoint
  - `POST /v1/chat/completions` — OpenAI-compatible multimodal chat endpoint
  - `GET /health` — Status and health check
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
    "image": "data:image/png;base64,...",
    "prompt": "Extract all information in markdown format.",
    "max_new_tokens": 4096
  }'
```
