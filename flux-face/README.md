# FLUX.1-schnell + InsightFace Face Swapper Server

A fast, zero-deformity thumbnail studio running on a remote Kaggle GPU.

### Architecture:
1. **Scene Generation**: FLUX.1-schnell (Q4_K_S GGUF) generates realistic 1024×576 16:9 thumbnail scenes in 4 steps (~8 seconds).
2. **Face Swapping**: InsightFace + `inswapper_128` swaps reference faces (e.g. Tarique Rahman and Narendra Modi) onto the generated people with 100% genuine likeness and zero facial deformity (~1 second).
3. **Total Pipeline Time**: ~10–12 seconds.

### Commands:
```bash
# Push kernel to Kaggle (DO NOT run until ready):
./bin/cluster flux-face up

# Generate a YouTube thumbnail:
./bin/cluster flux-face gen "A dramatic YouTube documentary thumbnail about Bangladesh military ambition, futuristic command center, tactical radar map screens, stealth fighter jets, cinematic lighting" \
  --face-left input-images/bangladesh-Tarique-Rahman-crop.jpg \
  --face-right input-images/India_PM_Narendra_Modi-2.jpeg \
  --size 1024x576 \
  -o thumbnail.png

# Stop kernel & release GPU:
./bin/cluster flux-face stop
```
