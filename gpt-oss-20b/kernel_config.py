"""Project config for the gpt-oss-20b server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "gpt-oss-20b",

    # --- model (downloaded ON KAGGLE) ---
    # Native MXFP4 build: ~12 GB, fits comfortably inside one T4 and leaves
    # plenty of room for context on the second.
    "__MODEL_REPO__": "ggml-org/gpt-oss-20b-GGUF",
    "__MODEL_FILE__": "gpt-oss-20b-MXFP4.gguf",
    "__MODEL_SIZE_GB__": "12.11",
    "__MODEL_ALIAS__": "gpt-oss",

    # --- llama.cpp ---
    "__LLAMA_BUILD__": "b11636",

    # --- runtime ---
    "__CTX__": "65536",          # native max is 131072
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    # gpt-oss uses the harmony format; llama.cpp needs a recent build for it.
    "__EXTRA_ARGS__": "",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- optional companions (leave "" to disable) ---
    # Eagle3 speculator: set "__DRAFT_FILE__": "eagle3-gpt-oss-20b-Q8_0.gguf"
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}