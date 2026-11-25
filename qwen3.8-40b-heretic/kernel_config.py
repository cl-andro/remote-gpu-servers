"""Project config for Qwen 3.8 40B Ultimate Heretic Uncensored vision-language server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Qwen 3.8 40B Heretic",
    "__MODEL_TYPE__": "llm",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "mradermacher/Qwen3.8-40B-Ultimate-Heretic-Uncensored-i1-GGUF",
    "__MODEL_FILE__": "Qwen3.8-40B-Ultimate-Heretic-Uncensored.i1-Q3_K_S.gguf",
    "__MODEL_SIZE_GB__": "17.5",
    "__MODEL_ALIAS__": "qwen3.8-40b-heretic",

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
    "__MMPROJ_REPO__": "mradermacher/Qwen3.8-40B-Ultimate-Heretic-Uncensored-GGUF",
    "__MMPROJ_FILE__": "mmproj-Qwen3.8-40B-Ultimate-Heretic-Uncensored-F16.gguf",
    "__MMPROJ_SIZE_GB__": "0.64",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}