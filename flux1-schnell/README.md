# FLUX.1-schnell — remote image generation from your PC

© 2026 CL-Andro / Mohammad Zaid — All Rights Reserved.

All source code, training datasets, documentation, and associated materials in
this repository are the exclusive intellectual property of CL-Andro / Mohammad
Zaid. Unauthorized copying, modification, redistribution, or commercial use — in
whole or in part — without prior written consent is strictly prohibited.

---

[← back to Remote GPU Servers](../README.md) · the **image generation** model of the
collection.

The model runs **on Kaggle** (GPU T4 x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

FLUX.1-schnell is Black Forest Labs' state-of-the-art **12B parameter flow-matching
diffusion transformer (MMDiT)** distilled for ultra-fast **1- to 4-step** generation.
This project runs the GGUF quantization from [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf)
using Hugging Face `diffusers` with GPU/CPU model offloading on Kaggle's dual T4 GPUs.

## Why this one

| Property | Value |
| --- | --- |
| Architecture | 12B Multimodal Diffusion Transformer (MMDiT) |
| Quantization | `flux1-schnell-Q4_K_S.gguf` (6.32 GB) from `city96` |
| Inference speed | 4 steps (guidance scale 0.0) — ~15–30s per image |
| VRAM fit | Fits inside 16 GB T4 with `enable_model_cpu_offload` |
| Precision | `torch.float16` compute (hardware-optimized for Tesla T4) |
| Text encoders | Dual encoders (CLIP-L + T5-XXL) |
| Web UI | Built-in interactive browser interface (zero setup) |
| API | OpenAI-compatible `/v1/images/generations` |
| License | Apache-2.0 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + generates a 512x512 sample image

# generate an image from terminal:
./ctr.sh gen "A futuristic cyberpunk city at midnight, neon reflections, 8k" -o city.png

# or open the URL in your browser for the full interactive Web UI:
cat url.txt
```

Startup takes a few minutes: downloads the ~6.3 GB GGUF transformer and pipeline components.

## Interactive Web UI

Opening the tunnel URL (`cat url.txt`) in any browser opens a self-contained Web UI:
- Multi-line prompt input with quick templates
- Resolution selector: `1024x1024`, `768x768`, `512x512`, `1024x768`, `768x1024`
- Step selector (1–4 steps) and optional seed
- Live progress feedback and instant image preview with one-click full-resolution download

## Commands

| Command | What it does |
| --- | --- |
| `./ctr.sh push` | Regenerate `notebook.ipynb` and start/restart the run |
| `./ctr.sh up` | `push`, then wait until the tunnel URL appears |
| `./ctr.sh url` | Scrape the live logs for the tunnel URL → `url.txt` |
| `./ctr.sh status` | Run state (`RUNNING` / `COMPLETE` / `ERROR`) |
| `./ctr.sh logs` / `follow` | Print recent logs / live-tail them |
| `./ctr.sh test` | Health check + generate sample test image |
| `./ctr.sh gen "prompt"` | Generate an image and save to file (`--out image.png`) |
| `./ctr.sh stop` | Stop the running session from your PC (frees the GPU) |
| `./ctr.sh output` | Download the run's output files |

### Stopping the session from your PC

Kaggle's public API token cannot cancel a session, so the notebook hosts a tiny
control server on a second tunnel and `./ctr.sh stop` calls it (secret random
path). The kernel polls the flag every 5 s; `stop` then waits for `COMPLETE`.

## OpenAI-Compatible API

The server implements the standard OpenAI Image Generation format:

```bash
curl -X POST "$(cat url.txt)/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic snow leopard on a Himalayan mountain peak, golden hour",
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

In Python with the `openai` SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url=open("url.txt").read().strip() + "/v1",
    api_key="not-needed"
)

response = client.images.generate(
    prompt="A cute astronaut cat floating in a galaxy with colorful nebulae",
    size="1024x1024",
    response_format="b64_json"
)

# save the base64 image
import base64
with open("cat.png", "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
```

## Configuration

Edit [`kernel_config.py`](kernel_config.py), then re-run `./ctr.sh push` (which
regenerates the notebook first).

| Key | Value | Notes |
| --- | --- | --- |
| `__MODEL_FILE__` | `flux1-schnell-Q4_K_S.gguf` | 6.32 GB; use `Q8_0` (11.8 GB) for higher quality |
| `__BASE_REPO__` | `Niansuh/FLUX.1-schnell` | Ungated repository for text encoders and VAE |
| `__DEFAULT_STEPS__` | `4` | Schnell generates in 1 to 4 steps |
| `__DEFAULT_WIDTH__` | `1024` | Default width |
| `__DEFAULT_HEIGHT__` | `1024` | Default height |
| `__SERVE_MINUTES__` | `660` | Kaggle caps sessions at 12 h |

## Notes and gotchas

- **4-step generation.** FLUX.1-schnell is designed specifically for 1–4 steps
  with guidance scale 0.0. Raising steps beyond 4 does not improve quality.
- **`torch.float16` compute.** Kaggle Tesla T4s do not have native hardware
  bfloat16 tensor cores. The pipeline runs in `torch.float16` for optimal speed.
- **CPU Offloading.** `pipe.enable_model_cpu_offload()` automatically manages VRAM,
  streaming components between CPU and GPU as each stage executes.
- **The tunnel URL is public.** Treat it as a shared secret and stop the session
  when you are done.
- **Quota:** a running GPU session consumes your weekly Kaggle GPU hours.
- UI fallback to stop: open `https://www.kaggle.com/code/zkmohammad/flux1-schnell-server`
  → *Stop Session*.
