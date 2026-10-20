"""Compute configuration supported by pinned omnivoice.cpp (src/backend.h)."""
from __future__ import annotations

BACKEND_ENV = "OMNIVOICE_GGUF_BACKEND"
DEVICE_ENV = "OMNIVOICE_GGUF_DEVICE"


def apply_compute_env(env: dict[str, str]) -> dict[str, str]:
    """Translate app overrides into GGML's native device-name environment."""
    from core.device_caps import requested_device_override

    if requested_device_override() == "cpu":
        env["GGML_BACKEND"] = "CPU"
        return env
    backend = env.get(BACKEND_ENV, "").strip().lower()
    index = env.get(DEVICE_ENV, "").strip()
    if not backend or backend == "auto":
        if index:
            raise ValueError(f"{DEVICE_ENV} requires {BACKEND_ENV}=cuda or vulkan")
        return env
    if backend not in {"cpu", "cuda", "vulkan"}:
        raise ValueError(f"{BACKEND_ENV} must be auto, cpu, cuda, or vulkan")
    if index and (not index.isascii() or not index.isdecimal()):
        raise ValueError(f"{DEVICE_ENV} must be a non-negative device index")
    if backend == "cpu" and index:
        raise ValueError(f"{DEVICE_ENV} is only supported with cuda or vulkan")
    env["GGML_BACKEND"] = (
        "CPU" if backend == "cpu" else
        ("CUDA" if backend == "cuda" else "Vulkan") + str(int(index or "0"))
    )
    return env


def runtime_profile(cls, caps) -> dict:
    """Do not infer a subprocess's devices from the parent's torch or loader.

    The pinned runtime has no capability enumeration API. Auto selection still
    uses native GPU backends, but its unverified route keeps the CPU timeout.
    """
    return {
        "gpu_compat": cls.gpu_compat,
        "min_vram_gb": float(getattr(cls, "min_vram_gb", 0) or 0),
        "effective_device": "cpu",
        "routing_status": "cpu_only",
        "routing_reason": "Native compute device unverified; using the CPU timeout budget",
        "runtime_backend": None,
        "runtime_device_index": None,
        "runtime_device_name": None,
        "runtime_hardware_family": None,
        "runtime_vram_gb": None,
        "runtime_device_verified": False,
    }
