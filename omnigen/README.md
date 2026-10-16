# OmniGen (Unified Multi-Image & Thumbnail Studio)

[OmniGen](https://github.com/staoxiao/OmniGen) by BAAI is a revolutionary open-source unified image generation model capable of:
1. **Multi-Subject Composition**: Upload 1, 2, or 3 reference portraits (e.g. Trump, Putin, Zelenskyy, or custom clients) and place them together in custom scenes while maintaining facial and subject consistency.
2. **Text-to-Image Generation**: Standard high-quality image generation from natural language prompts.
3. **In-Context Image Editing**: Modify existing images, change styles, or combine elements using prompt placeholders like `<img><|image_1|></img>` and `<img><|image_2|></img>`.

---

## Quick Start

### 1. Launch on Kaggle GPU
```bash
./bin/cluster omnigen up
```
This builds and pushes the notebook to Kaggle, loads OmniGen onto the GPU, opens your secure Cloudflare tunnel (`https://ai.883742.xyz`), and waits until the web UI is ready.

### 2. Interactive Web UI
Open your tunnel URL:
```
https://ai.883742.xyz
```
Log in with your `.env` credentials (`CLUSTER_AUTH_USER` / `CLUSTER_AUTH_PASS`).

- **Upload Box 1:** Reference Image 1 (maps to `<img><|image_1|></img>`)
- **Upload Box 2:** Reference Image 2 (maps to `<img><|image_2|></img>`)
- **Upload Box 3:** Reference Image 3 (maps to `<img><|image_3|></img>`)
- **Prompt:** Describe your scene, referencing subjects by tag:
  > *"A dramatic high-contrast YouTube thumbnail of <img><|image_1|></img> and <img><|image_2|></img> sitting across a geopolitical negotiation table, intense rim lighting, 8k photorealistic"*

### 3. CLI Image Generation
```bash
# Text-to-Image:
./bin/cluster omnigen gen "A dramatic cinematic YouTube thumbnail about a global conflict, 8k" -o thumb.png

# Multi-Image Composition:
./bin/cluster omnigen gen "A tense showdown thumbnail between <img><|image_1|></img> and <img><|image_2|></img>" \
  -i trump.jpg -i putin.jpg -o thumb.png
```

### 4. Stop when done
```bash
./bin/cluster omnigen stop
```
