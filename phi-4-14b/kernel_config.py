"""Project config for phi-4 server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "phi-4",
    "__MODEL_REPO__": "unsloth/phi-4-GGUF",
    "__MODEL_FILE__": "phi-4-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "8.28",
    "__MODEL_ALIAS__": "phi-4-14b",
    "__LLAMA_BUILD__": "b11636",
    "__CTX__": "16384",
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
