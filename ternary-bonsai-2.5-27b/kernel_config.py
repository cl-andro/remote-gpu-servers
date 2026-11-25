"""Project config for Ternary-Bonsai-2.5-27B (Prism ML) server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Ternary-Bonsai-2.5-27B",
    "__MODEL_TYPE__": "llm",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "prism-ml/Ternary-Bonsai-2.5-27B-GGUF",
    "__MODEL_FILE__": "Ternary-Bonsai-2.5-27B-Q2_0.gguf",
    "__MODEL_SIZE_GB__": "7.49",
    "__MODEL_ALIAS__": "ternary-bonsai-2.5-27b",

    # --- llama.cpp (b11556+ required for ternary Hadamard activation runtime) ---
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

    # --- optional vision projector (mmproj) ---
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}