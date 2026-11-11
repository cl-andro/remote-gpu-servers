"""Project config for Alibaba Qwen3.8-27B vision-language server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Qwen3.8-27B",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "unsloth/Qwen3.8-27B-GGUF",
    "__MODEL_FILE__": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "__MODEL_SIZE_GB__": "17.56",
    "__MODEL_ALIAS__": "qwen3.8-27b",

    # --- llama.cpp (b12174+ required for Qwen3.8 hybrid linear attention) ---
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
    "__MMPROJ_FILE__": "mmproj-BF16.gguf",
    "__MMPROJ_SIZE_GB__": "0.93",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
