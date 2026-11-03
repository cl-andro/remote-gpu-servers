"""Project config for Qwen3.5-4B-3D-Coder server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Qwen3.5-4B-3D-Coder",
    "__MODEL_REPO__": "ccore/Qwen3.5-4B-3D-Coder",
    "__MODEL_FILE__": "Qwen3.5-4B-3D-Coder-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "2.71",
    "__MODEL_ALIAS__": "qwen3.5-4b-3d-coder",
    "__LLAMA_BUILD__": "b11636",
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__MMPROJ_FILE__": "mmproj-Qwen3.5-4B-BF16.gguf",
    "__MMPROJ_SIZE_GB__": "0.68",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
