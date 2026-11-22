"""Project config for RadixArk DaisyRose-27B vision-language-reasoning server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "DaisyRose-27B",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "RadixArk/DaisyRose-27B-GGUF",
    "__MODEL_FILE__": "DaisyRose-27B-Q4_K_XL.gguf",
    "__MODEL_SIZE_GB__": "17.7",
    "__MODEL_ALIAS__": "daisyrose-27b",

    # --- llama.cpp (b12174+ required for Qwen3.8 hybrid linear attention) ---
    "__LLAMA_BUILD__": "b12174",

    # --- runtime ---
    "__CTX__": "65536",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "-b 512 -ub 512",
    "__FLASH_ATTN__": "off",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- vision / multimodal projector ---
    "__MMPROJ_FILE__": "mmproj-DaisyRose-27B-BF16.gguf",
    "__MMPROJ_SIZE_GB__": "0.9",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}