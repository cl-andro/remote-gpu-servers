"""Project config for DreamShaper 8 (SD 1.5) checkpoint server using stable-diffusion.cpp.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "DreamShaper 8",
    "__MODEL_TYPE__": "sd-checkpoint",

    # --- model (downloaded ON KAGGLE from HF) ---
    "__MODEL_REPO__": "Lykon/DreamShaper",
    "__MODEL_FILE__": "DreamShaper_8_pruned.safetensors",
    "__MODEL_SIZE_GB__": "2.1",
    "__MODEL_ALIAS__": "dreamshaper-8",

    # Optional VAE (leave empty to use model's built-in VAE)
    "__VAE_PATH__": "",
    "__LORA_PATH__": "",
    "__LORA_SCALE__": "1.0",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- default generation parameters ---
    "__DEFAULT_STEPS__": "20",
    "__DEFAULT_WIDTH__": "512",
    "__DEFAULT_HEIGHT__": "512",
}