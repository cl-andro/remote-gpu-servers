"""Project config for SDXL-Turbo server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "SDXL-Turbo",
    "__MODEL_TYPE__": "sdxl",
    "__MODEL_REPO__": "stabilityai/sdxl-turbo",
    "__MODEL_ALIAS__": "sdxl-turbo",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_STEPS__": "1",
    "__DEFAULT_WIDTH__": "512",
    "__DEFAULT_HEIGHT__": "512",
}
