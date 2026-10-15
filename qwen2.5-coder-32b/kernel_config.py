"""Project config for Qwen2.5-Coder-32B-Instruct server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Qwen2.5-Coder-32B-Instruct",
    "__MODEL_REPO__": "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",
    "__MODEL_FILE__": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "18.49",
    "__MODEL_ALIAS__": "qwen2.5-coder-32b",
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
