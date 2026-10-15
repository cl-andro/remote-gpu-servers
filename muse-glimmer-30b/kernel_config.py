"""Project config for the Muse-Glimmer-30B server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Muse-Glimmer-30B",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "meta-models/Muse-Glimmer-30B-GGUF",
    "__MODEL_FILE__": "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "16.76",
    "__MODEL_ALIAS__": "muse-glimmer",

    # --- llama.cpp (b10353+ required for the muse-glimmer architecture) ---
    "__LLAMA_BUILD__": "b11636",

    # --- runtime ---
    "__CTX__": "65536",          # keep the whole context for one request (-np 1)
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "--temp 1.0 --top-p 0.95 --top-k 64",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- optional companions (enabled by default for this model) ---
    # Perception encoder: enables image input.
    "__MMPROJ_FILE__": "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
    "__MMPROJ_SIZE_GB__": "1.4",
    # DFlash drafter: speculative decoding, ~1.6 GB, faster decode.
    "__DRAFT_FILE__": "dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
    "__DRAFT_SIZE_GB__": "1.63",
}