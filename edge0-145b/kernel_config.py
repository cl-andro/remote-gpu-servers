"""Edge0-145B (latent sparse MoE, Q3_K pack) — EXPERIMENTAL.

Not a llama.cpp/GGUF model: needs Edge0's Strata engine (CUDA, built from
source) and a ~105 GB asset pack. See README.md before pushing.
"""

CFG = {
    "__MODEL_TYPE__": "edge0",
    "__PROJECT_NAME__": "Edge0-145B",
    "__MODEL_REPO__": "Edge0/Edge0-145B-q3k",
    "__MODEL_FILE__": "model-skeleton.gguf",
    "__MODEL_SIZE_GB__": "5.99",
    "__MODEL_ALIAS__": "edge0-145b,edge0_145b",
    "__CTX__": "32768",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
}
