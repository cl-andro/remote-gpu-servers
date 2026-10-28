# NVIDIA Nemotron-3.5-Parse — Document OCR & Layout Intelligence

The model runs **on Kaggle** (GPU T4 x2 or T4 x1). Your PC only sends HTTP requests to a
public tunnel URL, so **nothing large is ever downloaded locally**.

> **Prerequisite:** copy `.env.example` to `.env` in the repo root and paste your
> Kaggle token there before running any `./ctr.sh` command.

NVIDIA Nemotron-3.5-Parse is an open-weights, high-speed document image parsing and optical character recognition (OCR) model (~631M parameters, sub-1B class). It combines an **NVIDIA C-RADIOv4-SO400M** vision encoder with an **mBART** causal prefix decoder to convert complex, unstructured documents—including multi-column research papers, tables, receipts, invoices, slides, and charts—into clean, structured Markdown and bounding boxes.

## Specifications

| Property | Value |
| --- | --- |
| Architecture | Multimodal Vision-Encoder-Decoder (C-RADIOv4 + mBART) |
| Parameters | ~631M (Sub-1B parameter class) |
| Model Repo | `nvidia/Nemotron-3.5-Parse` |
| Image Processing | Dynamic rectangular NaViT (supports native aspect ratios up to 1664x2048) |
| Output Modes | Clean Markdown, Layout Bounding Boxes, Semantic Classes, Charts & Tables |
| Hardware Requirements | Fits comfortably in single or dual Tesla T4 (~1.3 GB model footprint) |
| Web UI | Interactive Drag & Drop document parsing studio included |
| License | OpenMDW License v1.1 |

## Quick start

```bash
./ctr.sh push      # push the notebook to Kaggle and start the GPU run
./ctr.sh url       # wait for startup, then save the public URL to url.txt
./ctr.sh test      # health check + sample prompt test

# parse a document image from your terminal:
cluster nemotron-3.5-parse parse path/to/document.png

# save parsed markdown to file:
cluster nemotron-3.5-parse parse receipt.jpg --out receipt.md

# include bounding boxes and layout classes:
cluster nemotron-3.5-parse parse paper.png --mode full --raw

# open the web UI in browser:
# visit the URL printed by: cluster nemotron-3.5-parse url
```
