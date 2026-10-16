"""Project config for DeepSeek Janus-Pro-7B multimodal understanding & image generation server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Janus-Pro-7B",
    "__MODEL_TYPE__": "janus",
    "__MODEL_REPO__": "deepseek-ai/Janus-Pro-7B",
    "__MODEL_ALIAS__": "janus-pro-7b",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_CFG__": "5.0",
    "__DEFAULT_TEMP__": "1.0",
}
