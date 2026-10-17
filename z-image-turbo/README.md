# Z-Image-Turbo (8-Step Ultra-Fast Photorealistic & Thumbnail Studio)

[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) by Alibaba Cloud / Tongyi-MAI is an ultra-fast Scalable Single-Stream Diffusion Transformer ($\text{S}^3$-DiT) with 6 Billion parameters, distilled for exceptional quality in only **8 inference steps**:

1. **Ultra-Fast 8-Step Generation**: Produces crisp 1024×1024 photorealistic images in ~3–5 seconds on Kaggle Tesla T4 GPUs.
2. **Built-in World Leader & Celebrity Representation**: Deep native knowledge of global figures (Trump, Putin, Zelenskyy, Elon Musk, etc.) without requiring external LoRAs.
3. **High-Impact YouTube Thumbnails**: Excellent at dramatic lighting, facial expressions, cinematic depth of field, and bilingual typography rendering.
4. **Interactive Web UI & OpenAI-Compatible API**: Web UI with real-time preview and `/v1/images/generations` endpoint.

> **Tip:**
> - Use **Z-Image-Turbo** (`z-image-turbo`) for the fastest, most dramatic text-to-image thumbnails featuring world events and public figures.
> - Use **OmniGen** (`omnigen`) when you need to upload and fuse specific custom reference photos of people or products into a single scene.
> - Use **Janus-Pro-7B** (`janus-pro-7b`) for unified multimodal understanding and image generation.

---

## Quick Start

### 1. Launch on Kaggle GPU
```bash
./bin/cluster z-image-turbo up
```
This prepares and pushes the notebook to Kaggle, loads Z-Image-Turbo onto the GPU, opens your secure Cloudflare tunnel (`https://ai.883742.xyz`), and waits until the web UI is ready.

### 2. Interactive Web UI
Open your tunnel URL:
```
https://ai.883742.xyz
```
Log in with your `.env` credentials (`CLUSTER_AUTH_USER` / `CLUSTER_AUTH_PASS`).

- **Prompt Box**: Enter your scenario, e.g.:
  > *"A dramatic high-contrast YouTube thumbnail of Donald Trump, Vladimir Putin, and Volodymyr Zelenskyy seated around an intense geopolitical negotiation table in a modern war room, dramatic rim lighting, 8k photorealistic, cinematic"*
- **Resolution**: Select `1024x1024`, `1024x768` (16:9 thumbnail ratio), or `512x512` (fastest).
- **Steps**: Default 8 DiT steps (set to 9 in Diffusers schedule).
- Click **Generate Image** to preview and download full-resolution PNG.

### 3. CLI Image Generation
```bash
# Generate a YouTube thumbnail via terminal:
./bin/cluster z-image-turbo gen "A dramatic cinematic YouTube thumbnail of a high-stakes geopolitical debate between world leaders, neon rim light, 8k" -o thumb.png

# Custom resolution:
./bin/cluster z-image-turbo gen "Epic cinematic battle between futuristic armies, glowing energy shields, 8k" \
  --size 1024x768 -o epic.png
```

### 4. Stop when done
```bash
./bin/cluster z-image-turbo stop
```
