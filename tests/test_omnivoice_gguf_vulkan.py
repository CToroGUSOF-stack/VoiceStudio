"""Tests for OmniVoice GGUF Vulkan support and device overrides (Phase 1)."""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import patch

import pytest

from core.device_caps import HostCaps, UNUSABLE_GPU_MARKER, VULKAN_MARKER
from engines.omnivoice_gguf.backend import (
    BACKEND_ENV,
    DEVICE_ENV,
    OmniVoiceGGUFBackend,
)
from services.engine_routing import resolve_routing


def test_omnivoice_gguf_declares_vulkan():
    assert "vulkan" in OmniVoiceGGUFBackend.gpu_compat
    assert OmniVoiceGGUFBackend.gpu_compat == ("cuda", "vulkan", "mps", "cpu")


def test_compose_argv_passes_backend_and_device_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv(BACKEND_ENV, "vulkan")
    monkeypatch.setenv(DEVICE_ENV, "1")

    backend = OmniVoiceGGUFBackend()
    fake_bin = tmp_path / "omnivoice-tts"
    fake_bin.touch()
    fake_model = tmp_path / "model.gguf"
    fake_model.touch()
    fake_codec = tmp_path / "codec.gguf"
    fake_codec.touch()
    out_wav = tmp_path / "out.wav"

    with patch("engines.omnivoice_gguf.backend._binary_path", return_value=fake_bin):
        argv = backend._build_argv(
            base=fake_model,
            tokenizer=fake_codec,
            out_path=out_wav,
            ref_audio=None,
            ref_text=None,
            language="en",
        )

    assert "--backend" in argv
    assert argv[argv.index("--backend") + 1] == "vulkan"
    assert "--device" in argv
    assert argv[argv.index("--device") + 1] == "1"


def test_omnivoice_gguf_routes_to_vulkan_on_amd_windows_host():
    note = f"AMD Radeon RX 7900 XT (AMD) {UNUSABLE_GPU_MARKER}: this is a CUDA build"
    caps = HostCaps(
        family="cpu",
        available_families=("cpu",),
        notes=(note, f"{VULKAN_MARKER}; Vulkan-native engines can use the GPU"),
    )
    result = resolve_routing(OmniVoiceGGUFBackend.gpu_compat, caps)
    assert result["routing_status"] == "accelerated"
    assert result["effective_device"] == "vulkan"
