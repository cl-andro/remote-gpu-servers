"""Project config for the Devstral-Small-2-24B server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Devstral-Small-2-24B",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
    "__MODEL_FILE__": "Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf",
    "__MODEL_SIZE_GB__": "14.51",
    "__MODEL_ALIAS__": "devstral",

    # --- llama.cpp ---
    "__LLAMA_BUILD__": "b11636",

    # --- runtime ---
    "__CTX__": "65536",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- optional companions (leave "" to disable) ---
    # Vision projector: set "__MMPROJ_FILE__": "mmproj-F16.gguf"
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}