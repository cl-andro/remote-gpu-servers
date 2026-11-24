"""Project config for Inkling-2-Mini multimodal vision-language server (Thinking Machines Lab).

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Inkling-2-Mini",
    "__MODEL_TYPE__": "inkling-loop",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "unsloth/Inkling-2-Mini-GGUF",
    "__MODEL_FILE__": "Inkling-2-Mini-UD-Q4_K_XL.gguf",
    "__MODEL_SIZE_GB__": "17.6",
    "__MODEL_ALIAS__": "inkling-2-mini",

    # --- llama.cpp (b12174+ required for inkling2 looped architecture) ---
    "__LLAMA_BUILD__": "b12174",

    # --- runtime ---
    "__CTX__": "1000000",
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