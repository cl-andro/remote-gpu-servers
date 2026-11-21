"""Project config for Alibaba Qwen3.8-Flash-Next MoE server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Qwen3.8-Flash-Next",

    # --- model (downloaded ON KAGGLE) ---
    # Specialized 24GB footprint REAP25 UD-IQ4_XS quant: fits 100% in Kaggle dual T4 VRAM
    "__MODEL_REPO__": "anik-jha/Qwen3.8-Flash-Next-24GB-GGUF",
    "__MODEL_FILE__": "Qwen3.8-Flash-Next-24GB-REAP25-UD-IQ4_XS.gguf",
    "__MODEL_SIZE_GB__": "20.29",
    "__MODEL_ALIAS__": "qwen3.8-flash-next,Qwen3.8-Flash-Next",

    # --- llama.cpp (b12580+ required for Qwen4Exp / Qwen3.8-Flash-Next hybrid attention) ---
    "__LLAMA_BUILD__": "b12580",

    # --- N-gram lookup table (PLE REQUIRED for Qwen4Exp / Qwen3.8-Flash-Next) ---
    # Model file has NO n-gram table inside - MUST load via --ple or llama.cpp fails to load
    "__PLE_REPO__": "anik-jha/Qwen3.8-Flash-Next-24GB-GGUF",
    "__PLE_FILE__": "Qwen3.8-Flash-Next-ngram-table-IQ4_NL.gguf_file",
    "__PLE_SIZE_GB__": "28.80",

    # --- Recovery adapter (restores pruned expert outputs) ---
    "__ADAPTER_REPO__": "anik-jha/Qwen3.8-Flash-Next-24GB-GGUF",
    "__ADAPTER_FILE__": "recovery-adapter/Qwen3.8-Flash-Next-24GB-REAP25-adapter.gguf",
    "__ADAPTER_SIZE_GB__": "1.92",

    # --- runtime ---
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "-lm mmap -lzm on -ctk q8_0 -ctv q8_0 --presence-penalty 1.5 --repeat-last-n 20480 -b 2048 -ub 1024",
    "__FLASH_ATTN__": "on",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- vision / multimodal projector ---
    "__MMPROJ_REPO__": "",
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
