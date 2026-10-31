"""Project config for SenseNova-Loop-9B-Preview server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "SenseNova-Loop-9B-Preview",
    "__MODEL_TYPE__": "sensenova-loop",
    "__MODEL_REPO__": "sensenova/SenseNova-Loop-9B-Preview",
    "__MODEL_ALIAS__": "sensenova-loop-9b",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__CTX__": "32768",
    "__DEFAULT_MAX_TOKENS__": "4096",
}
