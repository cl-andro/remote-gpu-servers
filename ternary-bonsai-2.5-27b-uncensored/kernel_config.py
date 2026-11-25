"""Project config for Ternary-Bonsai-2.5-27B-Uncensored (AtomicChat) server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Ternary-Bonsai-2.5-27B-Uncensored",
    "__MODEL_TYPE__": "llm",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "AtomicChat/Ternary-Bonsai-2.5-27B-Uncensored-GGUF",
    "__MODEL_FILE__": "Ternary-Bonsai-2.5-27B-Uncensored-PTQ1_0.gguf",
    "__MODEL_SIZE_GB__": "5.99",
    "__MODEL_ALIAS__": "ternary-bonsai-2.5-27b-uncensored",

    # --- llama.cpp (PrismML fork required for PTQ1_0 ternary format) ---
    "__LLAMA_BUILD__": "prism-b12128-d7b2896",

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

    # --- optional vision projector (mmproj) ---
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}