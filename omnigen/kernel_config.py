"""Project config for OmniGen unified multi-image generation & thumbnail server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "OmniGen-v1",
    "__MODEL_TYPE__": "omnigen",
    "__MODEL_REPO__": "Shitao/OmniGen-v1",
    "__MODEL_ALIAS__": "omnigen",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_STEPS__": "30",
    "__DEFAULT_WIDTH__": "1024",
    "__DEFAULT_HEIGHT__": "1024",
}
