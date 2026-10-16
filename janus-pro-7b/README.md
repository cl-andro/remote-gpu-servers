# Janus-Pro-7B (DeepSeek Multimodal AI)

[Janus-Pro](https://github.com/deepseek-ai/Janus) by DeepSeek-AI is an advanced unified multimodal understanding and image generation model. 

- **Vision-Language Understanding**: Interprets visual content, analyzes scenes, and answers questions with DeepSeek intelligence.
- **Instruction-Driven Image Generation**: Generates intricate, high-detail visuals directly from text instructions with decoupled visual representations.

---

## Quick Start

### 1. Launch on Kaggle GPU
```bash
./bin/cluster janus-pro-7b up
```
This builds and pushes the notebook to Kaggle, loads the 7B model distributed across the dual Tesla T4 GPUs using `device_map="auto"`, and launches the secure Cloudflare tunnel (`https://ai.883742.xyz`).

### 2. Interactive Web UI
Visit your tunnel:
```
https://ai.883742.xyz
```
Log in with your `.env` username and password.

### 3. CLI Image Generation
```bash
./bin/cluster janus-pro-7b gen "A cinematic YouTube thumbnail of a high-tech robotic command center in 8k" -o thumb.png
```

### 4. Stop when done
```bash
./bin/cluster janus-pro-7b stop
```
