"""Project config for Ripple 65B Abliterated (OrcaRouter) server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Ripple 65B Abliterated",
    "__MODEL_TYPE__": "llm",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "orcarouter/Ripple-65B-abliterated-GGUF",
    "__MODEL_FILE__": "Ripple-65B-abliterated-Mix-3bit.gguf",
    "__MODEL_SIZE_GB__": "24.0",
    "__MODEL_ALIAS__": "ripple-65b-abliterated",

    # --- llama.cpp (needs PR #31692, in b12676+) ---
    "__LLAMA_BUILD__": "b12676",

    # --- runtime ---
    "__CTX__": "131072",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "-b 512 -ub 512 -fa on",
    "__FLASH_ATTN__": "on",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- no vision projector ---
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}