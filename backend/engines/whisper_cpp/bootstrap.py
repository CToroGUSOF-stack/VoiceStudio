"""whisper.cpp binary probe and model resolution.

whisper.cpp (ggerganov/whisper.cpp) is a high-performance C/C++ implementation
of OpenAI's Whisper model, supporting GGML/GGUF quantization and native GPU
backends including Vulkan, CUDA, and Metal.

Probe order for the binary:
    1. ${OMNIVOICE_WHISPER_CPP_BIN} — absolute path to the binary itself.
    2. ${OMNIVOICE_WHISPER_CPP_DIR}/whisper-cli[.exe] — user-managed install dir.
    3. backend/engines/whisper_cpp/bin/whisper-cli[.exe] — package-local copy.
    4. System PATH lookup via shutil.which().

Local-first: no automatic network downloads of executable binaries.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("omnivoice.whisper_cpp.bootstrap")

#: Pinned default model identifier
DEFAULT_MODEL_NAME = "ggml-base.bin"

#: Hugging Face repo serving the GGML/GGUF models
HF_MODEL_REPO = "ggerganov/whisper.cpp"

#: Environment variable overrides
BIN_ENV = "OMNIVOICE_WHISPER_CPP_BIN"
DIR_ENV = "OMNIVOICE_WHISPER_CPP_DIR"
MODEL_ENV = "OMNIVOICE_WHISPER_CPP_MODEL"
BACKEND_ENV = "OMNIVOICE_WHISPER_CPP_BACKEND"
DEVICE_ENV = "OMNIVOICE_WHISPER_CPP_DEVICE"

#: Package-local binary and model directories
_PKG_DIR = Path(__file__).resolve().parent
_PKG_BIN_DIR = _PKG_DIR / "bin"
_PKG_MODELS_DIR = _PKG_DIR / "models"

#: Candidate binary names
_CANDIDATE_NAMES = (
    ("whisper-cli.exe", "whisper-server.exe", "main.exe", "server.exe", "whisper.exe")
    if sys.platform == "win32"
    else ("whisper-cli", "whisper-server", "main", "server", "whisper")
)


def locate_binary() -> Optional[Path]:
    """Find a usable whisper.cpp binary.

    Returns the absolute Path if found, otherwise None.
    """
    # 1. Direct environment variable override
    direct = os.environ.get(BIN_ENV, "").strip()
    if direct:
        cand = Path(direct)
        if cand.is_file():
            return cand.resolve()
        logger.warning("%s points to non-existent file: %s", BIN_ENV, direct)

    # 2. Directory override
    dir_env = os.environ.get(DIR_ENV, "").strip()
    if dir_env:
        dir_path = Path(dir_env)
        if dir_path.is_dir():
            for name in _CANDIDATE_NAMES:
                cand = dir_path / name
                if cand.is_file():
                    return cand.resolve()

    # 3. Package-owned binary directory
    if _PKG_BIN_DIR.is_dir():
        for name in _CANDIDATE_NAMES:
            cand = _PKG_BIN_DIR / name
            if cand.is_file():
                return cand.resolve()

    # 4. System PATH
    for name in _CANDIDATE_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    return None


def resolve_model(model_name: Optional[str] = None) -> Optional[Path]:
    """Resolve the path to a GGML/GGUF Whisper model file.

    Checks:
    1. Direct file path if model_name (or MODEL_ENV) points to an existing file.
    2. Package-local models directory: `backend/engines/whisper_cpp/models/`.
    3. User whisper cache: `~/.cache/whisper/`.
    4. HuggingFace hub cache if present.
    """
    target = (model_name or os.environ.get(MODEL_ENV, "")).strip() or DEFAULT_MODEL_NAME

    # Direct existing file path
    direct = Path(target)
    if direct.is_file():
        return direct.resolve()

    filename = direct.name if direct.suffix in (".bin", ".gguf") else f"ggml-{target}.bin"

    # In package models dir
    pkg_cand = _PKG_MODELS_DIR / filename
    if pkg_cand.is_file():
        return pkg_cand.resolve()

    # In user cache dir
    cache_cand = Path.home() / ".cache" / "whisper" / filename
    if cache_cand.is_file():
        return cache_cand.resolve()

    # In huggingface hub cache (safe check, no network)
    try:
        from huggingface_hub import try_to_load_from_cache

        hf_cached = try_to_load_from_cache(HF_MODEL_REPO, filename)
        if hf_cached and isinstance(hf_cached, (str, Path)) and Path(hf_cached).is_file():
            return Path(hf_cached).resolve()
    except (ImportError, Exception):  # noqa: BLE001
        pass

    return None


def is_installed() -> bool:
    """Return True if a usable whisper.cpp binary is located."""
    return locate_binary() is not None
