"""Project config for the FLUX.1-schnell + InsightFace Face Swapper server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "FLUX.1-schnell + Face Swapper",
    "__MODEL_TYPE__": "flux-face",

    # --- model (downloaded ON KAGGLE) ---
    "__MODEL_REPO__": "city96/FLUX.1-schnell-gguf",
    "__MODEL_FILE__": "flux1-schnell-Q4_K_S.gguf",
    "__MODEL_SIZE_GB__": "6.32",
    "__MODEL_ALIAS__": "flux-face",

    # --- base pipeline components (text encoders, tokenizer, VAE - ungated) ---
    "__BASE_REPO__": "Niansuh/FLUX.1-schnell",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- default generation parameters (16:9 YouTube thumbnail default) ---
    "__DEFAULT_STEPS__": "4",
    "__DEFAULT_WIDTH__": "1024",
    "__DEFAULT_HEIGHT__": "576",
}
