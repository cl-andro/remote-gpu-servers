"""Project config for the Qwen3-Coder-30B-A3B agent server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Qwen3-Coder-30B-A3B",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "__MODEL_FILE__": "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
    "__MODEL_SIZE_GB__": "17.67",
    "__MODEL_ALIAS__": "qwen3-coder",

    # --- llama.cpp (b10353+ required for the newest architectures) ---
    "__LLAMA_BUILD__": "b11636",

    # --- runtime ---
    "__CTX__": "65536",          # native max is 262144
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "",        # Qwen3-Coder is a non-thinking model

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- optional companions (leave "" to disable) ---
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}