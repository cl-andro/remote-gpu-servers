"""Project config for Baidu Unlimited-OCR-2 document intelligence server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Unlimited-OCR-2",
    "__MODEL_TYPE__": "unlimited-ocr-2",
    "__MODEL_REPO__": "baidu/Unlimited-OCR-2",
    "__MODEL_ALIAS__": "unlimited-ocr-2",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_MAX_TOKENS__": "4096",
}
