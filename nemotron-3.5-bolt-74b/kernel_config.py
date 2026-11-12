"""Project config for NVIDIA Nemotron-3.5-Bolt-74B-A9B vision-language MoE server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Nemotron-3.5-Bolt-74B-A9B",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF",
    "__MODEL_FILE__": "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf",
    "__MODEL_SIZE_GB__": "20.51",
    "__MODEL_ALIAS__": "nemotron-3.5-bolt",

    # --- llama.cpp (b12174+ required for Nemotron-3.5 Mamba2-Transformer MoE) ---
    "__LLAMA_BUILD__": "b12174",

    # --- runtime ---
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "-b 512 -ub 512",
    "__FLASH_ATTN__": "off",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- vision / multimodal projector ---
    "__MMPROJ_FILE__": "mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf",
    "__MMPROJ_SIZE_GB__": "1.61",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
