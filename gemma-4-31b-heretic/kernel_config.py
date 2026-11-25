"""Project config for Gemma 4 31B Heretic (uncensored) vision-language server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Gemma 4 31B Heretic",
    "__MODEL_TYPE__": "llm",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "mradermacher/gemma-4-31B-it-heretic-i1-GGUF",
    "__MODEL_FILE__": "gemma-4-31B-it-heretic.i1-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "18.8",
    "__MODEL_ALIAS__": "gemma-4-31b-heretic",

    # --- llama.cpp ---
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

    # --- vision projector ---
    "__MMPROJ_REPO__": "mradermacher/gemma-4-31B-it-heretic-GGUF",
    "__MMPROJ_FILE__": "mmproj-gemma-4-31B-it-heretic-F16.gguf",
    "__MMPROJ_SIZE_GB__": "0.64",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}