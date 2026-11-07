"""Project config for Xiaomi-OCR-0 document intelligence server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Xiaomi-OCR-0",
    "__MODEL_TYPE__": "xiaomi-ocr-0",
    "__MODEL_REPO__": "xiaomi-research/Xiaomi-OCR-0",
    "__MODEL_ALIAS__": "xiaomi-ocr-0",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_MAX_TOKENS__": "4096",
}
