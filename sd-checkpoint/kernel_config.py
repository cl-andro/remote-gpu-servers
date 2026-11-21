"""Project config for generic Stable Diffusion Checkpoint server using stable-diffusion.cpp.

This project loads ANY .safetensors or .ckpt checkpoint from Hugging Face.
Configure the model via environment variables or edit this file.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push

Usage examples:
  # Use DreamShaper 8 (SD 1.5)
  __MODEL_REPO__="Lykon/DreamShaper" __MODEL_FILE__="DreamShaper_8_pruned.safetensors" __MODEL_SIZE_GB__="2.1" ./ctr.sh push

  # Use CyberRealistic V8 (SD 1.5)
  __MODEL_REPO__="DSchwabe/CyberRealistic" __MODEL_FILE__="CyberRealistic_V8_FP16.safetensors" __MODEL_SIZE_GB__="2.0" ./ctr.sh push

  # Use Juggernaut XL v9 (SDXL)
  __MODEL_REPO__="RunDiffusion/Juggernaut-XL-v9" __MODEL_FILE__="Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors" __MODEL_SIZE_GB__="7.1" ./ctr.sh push

  # Use any custom checkpoint
  __MODEL_REPO__="your-username/your-model" __MODEL_FILE__="your-model.safetensors" __MODEL_SIZE_GB__="X.X" ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "SD Checkpoint (Generic)",
    "__MODEL_TYPE__": "sd-checkpoint",

    # --- model (downloaded ON KAGGLE from HF) ---
    # Override these via environment variables:
    #   __MODEL_REPO__="Lykon/DreamShaper"
    #   __MODEL_FILE__="DreamShaper_8_pruned.safetensors"
    #   __MODEL_SIZE_GB__="2.1"
    "__MODEL_REPO__": "Lykon/DreamShaper",
    "__MODEL_FILE__": "DreamShaper_8_pruned.safetensors",
    "__MODEL_SIZE_GB__": "2.1",
    "__MODEL_ALIAS__": "sd-checkpoint",

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