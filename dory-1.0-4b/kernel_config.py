"""Project config for NVIDIA Nemotron Labs Dory 1.0 4B server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Dory1.0-4B",
    "__MODEL_REPO__": "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF",
    "__MODEL_FILE__": "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf",
    "__MODEL_SIZE_GB__": "3.42",
    "__MODEL_ALIAS__": "dory-1.0-4b",
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
