#!/usr/bin/env python3
"""Generic Kaggle kernel generator for the remote-GPU inference projects.

Usage
-----
    python3 lib/make_kernel.py <project-dir>

``<project-dir>`` must contain a ``kernel_config.py`` defining a ``CFG`` dict
whose keys are the ``__PLACEHOLDER__`` tokens documented in ``lib/cells.py``.
The generator writes ``<project-dir>/notebook.ipynb``.

Everything heavy (model download, llama.cpp, inference) happens ON KAGGLE.
Nothing is downloaded to the local machine.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cells import CELLS, FLUX_CELLS, SDXL_CELLS  # noqa: E402

# Placeholders whose value is not allowed to be empty.
REQUIRED_LLM = (
    "__PROJECT_NAME__", "__MODEL_REPO__", "__MODEL_FILE__", "__MODEL_SIZE_GB__",
    "__LLAMA_BUILD__", "__CTX__", "__NGL__", "__PORT__", "__CTL_PORT__",
    "__SERVE_MINUTES__", "__MODEL_ALIAS__",
)

DEFAULTS_LLM = {
    "__MMPROJ_FILE__": "", "__MMPROJ_SIZE_GB__": "",
    "__DRAFT_FILE__": "", "__DRAFT_SIZE_GB__": "",
    "__SPLIT_MODE__": "layer", "__EXTRA_ARGS__": "",
}

REQUIRED_FLUX = (
    "__PROJECT_NAME__", "__MODEL_REPO__", "__MODEL_FILE__", "__MODEL_SIZE_GB__",
    "__BASE_REPO__", "__PORT__", "__CTL_PORT__",
    "__SERVE_MINUTES__", "__MODEL_ALIAS__",
)

DEFAULTS_FLUX = {
    "__DEFAULT_STEPS__": "4",
    "__DEFAULT_WIDTH__": "1024",
    "__DEFAULT_HEIGHT__": "1024",
}

REQUIRED_SDXL = (
    "__PROJECT_NAME__", "__MODEL_REPO__", "__PORT__", "__CTL_PORT__",
    "__SERVE_MINUTES__", "__MODEL_ALIAS__",
)

DEFAULTS_SDXL = {
    "__DEFAULT_STEPS__": "1",
    "__DEFAULT_WIDTH__": "512",
    "__DEFAULT_HEIGHT__": "512",
    "__MODEL_FILE__": "sdxl-turbo",
    "__MODEL_SIZE_GB__": "6.0",
}


def load_cfg(project_dir: str) -> dict:
    path = os.path.join(project_dir, "kernel_config.py")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location("kernel_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = dict(getattr(mod, "CFG", {}))

    model_type = cfg.get("__MODEL_TYPE__", "llm")
    if model_type == "flux":
        defaults = DEFAULTS_FLUX
        required = REQUIRED_FLUX
    elif model_type == "sdxl":
        defaults = DEFAULTS_SDXL
        required = REQUIRED_SDXL
    else:
        defaults = DEFAULTS_LLM
        required = REQUIRED_LLM

    for key, val in defaults.items():
        cfg.setdefault(key, val)
    # Allow overriding any config key via environment variable (e.g. __MODEL_FILE__=...)
    for key in set(required) | set(defaults):
        if key in os.environ:
            cfg[key] = os.environ[key]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise SystemExit(f"{path}: missing CFG keys: {', '.join(missing)}")
    return cfg


def render(src: str, cfg: dict) -> str:
    for key, val in cfg.items():
        src = src.replace(key, str(val))
    return src


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    project_dir = os.path.abspath(sys.argv[1])
    cfg = load_cfg(project_dir)

    model_type = cfg.get("__MODEL_TYPE__", "llm")
    if model_type == "flux":
        template_cells = FLUX_CELLS
    elif model_type == "sdxl":
        template_cells = SDXL_CELLS
    else:
        template_cells = CELLS
    cells = [render(c, cfg) for c in template_cells]

    # Fail fast if any placeholder is left unsubstituted.
    for i, c in enumerate(cells):
        if "__" in c:
            leftovers = sorted({t for t in c.split() if t.startswith("__") and t.endswith("__")})
            if leftovers:
                raise SystemExit(f"cell {i}: unsubstituted placeholders {leftovers}")
        compile(c, f"<cell{i}>", "exec")

    nb = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": [(ln + "\n") for ln in c.strip("\n").split("\n")]}
            for c in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = os.path.join(project_dir, "notebook.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out} ({len(cells)} cells) for {cfg['__PROJECT_NAME__']}")


if __name__ == "__main__":
    main()