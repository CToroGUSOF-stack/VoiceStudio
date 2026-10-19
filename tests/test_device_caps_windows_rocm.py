"""Tests for Windows ROCm/HIP diagnostics in device_caps.py (Phase 2)."""
from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from core.device_caps import why_no_gpu


def _torch_rocm_mock(hip="6.4"):
    return types.SimpleNamespace(
        version=types.SimpleNamespace(hip=hip, cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )


def test_windows_rocm_missing_hip_dll_gives_sdk_advice(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    torch = _torch_rocm_mock()
    notes = why_no_gpu(torch)
    assert len(notes) == 1
    assert "amdhip64.dll was not found" in notes[0]
    assert "AMD HIP SDK for Windows" in notes[0]


def test_windows_rocm_with_hip_dll_gives_general_rocm_advice(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.delenv("HIP_PATH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "C:\\Windows\\System32\\amdhip64.dll")

    torch = _torch_rocm_mock()
    notes = why_no_gpu(torch)
    assert len(notes) == 1
    assert "ROCm 6.4 is installed on Windows but no GPU was enumerated" in notes[0]


def test_windows_rocm_with_override_gives_override_advice(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    monkeypatch.setattr("shutil.which", lambda name: "C:\\Windows\\System32\\amdhip64.dll")

    torch = _torch_rocm_mock()
    notes = why_no_gpu(torch)
    assert len(notes) == 1
    assert "HSA_OVERRIDE_GFX_VERSION=11.0.0" in notes[0]
