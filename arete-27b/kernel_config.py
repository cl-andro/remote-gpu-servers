"""Project config for NousResearch Arete-27B server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Arete-27B",
    "__MODEL_REPO__": "bartowski/NousResearch_Arete-27B-GGUF",
    "__MODEL_FILE__": "NousResearch_Arete-27B-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "16.02",
    "__MODEL_ALIAS__": "arete-27b",
    "__LLAMA_BUILD__": "b12174",
    "__CTX__": "32768",
    "__NGL__": "99",
    "__SPLIT_MODE__": "layer",
    "__EXTRA_ARGS__": "",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__MMPROJ_FILE__": "mmproj-NousResearch_Arete-27B-bf16.gguf",
    "__MMPROJ_SIZE_GB__": "0.87",
    "__DRAFT_FILE__": "",
    "__DRAFT_SIZE_GB__": "",
}
