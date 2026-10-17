"""Project config for Z-Image-Turbo 8-step ultra-fast image generation & thumbnail server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Z-Image-Turbo",
    "__MODEL_TYPE__": "zimage",
    "__MODEL_REPO__": "Tongyi-MAI/Z-Image-Turbo",
    "__MODEL_ALIAS__": "z-image-turbo",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_STEPS__": "9",
    "__DEFAULT_WIDTH__": "1024",
    "__DEFAULT_HEIGHT__": "1024",
}
