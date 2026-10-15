"""Project config for MiniCPM-V-2_6 server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "MiniCPM-V-2_6",
    "__MODEL_REPO__": "openbmb/MiniCPM-V-2_6-gguf",
    "__MODEL_FILE__": "ggml-model-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "4.36",
    "__MODEL_ALIAS__": "minicpm-v-2_6",
    "__LLAMA_BUILD__": "b11636",
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__MMPROJ_FILE__": "mmproj-model-f16.gguf",
    "__MMPROJ_SIZE_GB__": "0.97",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
