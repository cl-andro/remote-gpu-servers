# Ming-3.0-TTS (0.5B) — Studio-Grade Neural Text-to-Speech

The model runs **on Kaggle** (GPU T4 x1 or x2). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

Ming-3.0-TTS-0.5B is an open-source neural text-to-speech model developed by **InclusionAI (Ant Group)** under the MIT license. Powered by a Qwen2.5-0.5B dense backbone and an AudioVAE flow-matching vocoder, it delivers **44.1 kHz studio-grade audio**, zero-shot voice cloning, and inline paralinguistic expression control (`[laughter]`, `[breath]`, `[sigh]`, `[whisper]`, `[cough]`).

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Dense Qwen2.5-0.5B Backbone + CAM++ Speaker Embedding + AudioVAE Flow Matching |
| Parameters | ~0.5B parameters (~1.2 GB download footprint) |
| Model Repo | `inclusionAI/Ming-3.0-tts-0.5B` |
| Audio Quality | 44.1 kHz mono WAV |
| Languages | English, Chinese, Cantonese, Sichuanese, Spanish, French, German, Japanese, Korean |
| Capabilities | Text-to-Speech, Zero-Shot Voice Cloning, Paralinguistic Expression Tags |
| Web Studio | Built-in interactive studio with audio player and inline expression tags |
| API Endpoints | OpenAI-compatible `/v1/audio/speech` + custom `/v1/tts` |
| License | MIT License |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# synthesize speech from terminal:
cluster ming-3.0-tts-0.5b tts "Hello world, this is Ming 3.0 speaking on Kaggle GPU." --out speech.wav

# synthesize with another language:
cluster ming-3.0-tts-0.5b tts "Bonjour tout le monde, bienvenue." --language French --out french.wav

# zero-shot voice cloning with reference audio:
cluster ming-3.0-tts-0.5b tts "Cloning this speaker voice." --prompt-audio sample.wav --prompt-text "Sample transcript" --out cloned.wav

# open interactive Web Studio in browser:
# visit the URL printed by: cluster ming-3.0-tts-0.5b url
```
