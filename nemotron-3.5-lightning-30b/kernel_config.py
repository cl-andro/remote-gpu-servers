"""Project config for NVIDIA Nemotron-3.5-Lightning-30B-A3B server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B",

    # --- model (downloaded ON KAGGLE) ---
    # Official ggml-org conversion of the BF16 reference weights.
    "__MODEL_REPO__": "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
    "__MODEL_FILE__": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf",
    "__MODEL_SIZE_GB__": "18.90",
    "__MODEL_ALIAS__": "nemotron-3.5-lightning,nemotron-3.5-lightning-30b,nemotron-3.5,nemotron-3.5-30b",

    # --- llama.cpp (b12360+ required for the nemotron_h_moe architecture) ---
    "__LLAMA_BUILD__": "b12360",

    # --- runtime ---
    # 6 attention layers + 26 Mamba-2 layers, so the KV/SSM cache is tiny and
    # the weights dominate VRAM. 32k context is comfortable on dual T4s.
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "-b 512 -ub 512",
    "__FLASH_ATTN__": "off",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- optional companions ---
    # The repo also ships mtp-*.gguf drafters for speculative decoding; leave
    # empty for maximum stability (enable later via __DRAFT_FILE__=... ./ctr.sh push).
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
