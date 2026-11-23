"""Project config for SenseNova-Loop-9B multimodal vision-language server (Release).

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "SenseNova-Loop-9B",
    "__MODEL_TYPE__": "sensenova-loop",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "sensenova/SenseNova-Loop-9B-GGUF",
    "__MODEL_FILE__": "SenseNova-Loop-9B-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "5.5",
    "__MODEL_ALIAS__": "sensenova-loop-9b",

    # --- llama.cpp (b12174+ required for looped architecture) ---
    "__LLAMA_BUILD__": "b12174",

    # --- runtime ---
    "__CTX__": "524288",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "-b 512 -ub 512",
    "__FLASH_ATTN__": "off",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- vision / multimodal projector ---
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}