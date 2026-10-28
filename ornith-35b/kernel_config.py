"""Project config for Ornith-1.5-35B server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Ornith-1.5-35B",
    "__MODEL_REPO__": "bartowski/Ornith-1.5-35B-A3B-GGUF",
    "__MODEL_FILE__": "Ornith-1.5-35B-A3B-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "20.36",
    "__MODEL_ALIAS__": "ornith-35b",
    "__LLAMA_BUILD__": "b11636",
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__MMPROJ_FILE__": "",
    "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
