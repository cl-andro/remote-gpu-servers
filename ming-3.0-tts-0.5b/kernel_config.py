"""Project config for Ant Group Ming-3.0-TTS 0.5B studio server.

The shared generator (../lib/make_kernel.py) reads `CFG` and writes
`notebook.ipynb`. Edit values here, then run:  ./ctr.sh push
"""

CFG = {
    "__PROJECT_NAME__": "Ming-3.0-TTS-0.5B",
    "__MODEL_TYPE__": "ming-tts",
    "__MODEL_REPO__": "inclusionAI/Ming-3.0-tts-0.5B",
    "__MODEL_ALIAS__": "ming-3.0-tts-0.5b",
    "__PORT__": "8080",
    "__CTL_PORT__": "8081",
    "__SERVE_MINUTES__": "660",
    "__DEFAULT_LANGUAGE__": "English",
}
