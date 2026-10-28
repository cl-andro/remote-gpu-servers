"""Project config for NVIDIA Nemotron-3.5-Parse document OCR & layout server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Nemotron-3.5-Parse",
    "__MODEL_TYPE__": "parse",
    "__MODEL_REPO__": "nvidia/Nemotron-3.5-Parse",
    "__MODEL_ALIAS__": "nemotron-3.5-parse",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_MAX_TOKENS__": "4096",
}
