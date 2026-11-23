"""Project config for CCORE Web Coder 4B server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "CCORE Web Coder 4B",
    "__MODEL_TYPE__": "llm",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "ccore/CCORE-Web-Coder-4B-GGUF",
    "__MODEL_FILE__": "ccore-web-coder-Q8_0.gguf",
    "__MODEL_SIZE_GB__": "4.48",
    "__MODEL_ALIAS__": "ccore-web-coder-4b",

    # --- llama.cpp ---
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

    # --- optional vision projector (mmproj) ---
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}