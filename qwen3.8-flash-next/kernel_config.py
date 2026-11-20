"""Project config for Alibaba Qwen3.8-Flash-Next MoE server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Qwen3.8-Flash-Next",

    # --- model (downloaded ON KAGGLE) ---
    # Specialized 24GB footprint REAP25 UD-IQ4_XS quant: fits 100% in Kaggle dual T4 VRAM
    "__MODEL_REPO__": "anik-jha/Qwen3.8-Flash-Next-24GB-GGUF",
    "__MODEL_FILE__": "Qwen3.8-Flash-Next-24GB-REAP25-UD-IQ4_XS.gguf",
    "__MODEL_SIZE_GB__": "18.89",
    "__MODEL_ALIAS__": "qwen3.8-flash-next,Qwen3.8-Flash-Next",

    # --- llama.cpp (b12174+ required for Qwen4Exp / Qwen3.8-Flash-Next hybrid attention) ---
    "__LLAMA_BUILD__": "b12174",

    # --- runtime ---
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "-b 2048 -ub 1024",
    "__FLASH_ATTN__": "on",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- vision / multimodal projector (from unsloth official repo) ---
    "__MMPROJ_REPO__": "unsloth/Qwen3.8-Flash-Next-GGUF",
    "__MMPROJ_FILE__": "mmproj-BF16.gguf",
    "__MMPROJ_SIZE_GB__": "0.85",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
