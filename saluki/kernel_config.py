"""Project config for the Underdog-Saluki-27B server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Underdog-Saluki-27B",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "ConwayResearch/Underdog-Saluki-27B-1.0",
    "__MODEL_FILE__": "Underdog-Saluki-27B-1.0-IQ2-mix.gguf",
    "__MODEL_SIZE_GB__": "7.89",
    "__MODEL_ALIAS__": "saluki",

    # --- llama.cpp (must be >= b10330 for the qwen35 architecture) ---
    "__LLAMA_BUILD__": "b11636",

    # --- runtime ---
    "__CTX__": "32768",          # native max is 262144
    "__NGL__": "99",             # offload every layer
    "__SPLIT_MODE__": "layer",   # shard layers across both T4s
    "__EXTRA_ARGS__": "",

    # --- networking ---
    "__PORT__": "8080",          # llama-server (chat tunnel target)
    "__CTL_PORT__": "8081",      # control server (stop tunnel target)
    "__SERVE_MINUTES__": "660",  # Kaggle caps a session at 12 h

    # --- optional companions (leave "" to disable) ---
    "__MMPROJ_FILE__": "",       # "mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf" for vision
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}