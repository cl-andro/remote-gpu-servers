"""Project config for Juggernaut XL v9 Lightning (SDXL) checkpoint server using stable-diffusion.cpp.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Juggernaut XL v9 Lightning",
    "__MODEL_TYPE__": "sd-checkpoint",

    # --- model (downloaded ON KAGGLE from HF) ---
    "__MODEL_REPO__": "RunDiffusion/Juggernaut-XL-v9",
    "__MODEL_FILE__": "Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors",
    "__MODEL_SIZE_GB__": "7.1",
    "__MODEL_ALIAS__": "juggernaut-xl-v9",

    # Optional VAE (leave empty to use model's built-in VAE)
    "__VAE_PATH__": "",
    "__LORA_PATH__": "",
    "__LORA_SCALE__": "1.0",

    # --- networking ---
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",

    # --- default generation parameters ---
    "__DEFAULT_STEPS__": "4",
    "__DEFAULT_WIDTH__": "1024",
    "__DEFAULT_HEIGHT__": "1024",
}