"""Project config for Hermes-3-Llama-3.1-8B server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Hermes-3-Llama-3.1-8B",
    "__MODEL_REPO__": "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
    "__MODEL_FILE__": "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "4.58",
    "__MODEL_ALIAS__": "hermes-3-8b",
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
