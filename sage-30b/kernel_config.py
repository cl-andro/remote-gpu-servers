"""Project config for Sage-A1.5-30B server.
The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Sage-A1.5-30B",
    "__MODEL_REPO__": "InternScience/Sage-A1.5-Q4_K_M-GGUF",
    "__MODEL_FILE__": "Sage-A1.5-Q4_K_M.gguf",
    "__MODEL_SIZE_GB__": "20.22",
    "__MODEL_ALIAS__": "sage-30b",
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
