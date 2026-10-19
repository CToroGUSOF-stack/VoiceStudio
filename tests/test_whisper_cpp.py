"""Unit tests for whisper.cpp ASR backend with Vulkan support (Phase 3)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.device_caps import HostCaps, VULKAN_MARKER
from engines.whisper_cpp import (
    BACKEND_ENV,
    BIN_ENV,
    DEVICE_ENV,
    DIR_ENV,
    MODEL_ENV,
    WhisperCppBackend,
    is_installed,
    locate_binary,
    resolve_model,
)
from services import asr_backend as ab
from services.engine_routing import resolve_routing


# ── Binary and Model Discovery Tests ──────────────────────────────────────────


def test_locate_binary_via_bin_env(tmp_path, monkeypatch):
    fake_bin = tmp_path / "whisper-cli.exe"
    fake_bin.write_text("binary content")
    monkeypatch.setenv(BIN_ENV, str(fake_bin))

    assert locate_binary() == fake_bin.resolve()


def test_locate_binary_via_dir_env(tmp_path, monkeypatch):
    monkeypatch.delenv(BIN_ENV, raising=False)
    fake_bin = tmp_path / "whisper-cli"
    fake_bin.write_text("binary content")
    monkeypatch.setenv(DIR_ENV, str(tmp_path))

    # Test candidate lookup
    monkeypatch.setattr("engines.whisper_cpp.bootstrap._CANDIDATE_NAMES", ("whisper-cli",))
    assert locate_binary() == fake_bin.resolve()


def test_locate_binary_via_path(monkeypatch):
    monkeypatch.delenv(BIN_ENV, raising=False)
    monkeypatch.delenv(DIR_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/whisper-cli")
    monkeypatch.setattr("engines.whisper_cpp.bootstrap._PKG_BIN_DIR", Path("/nonexistent/bin"))

    assert locate_binary() == Path("/usr/local/bin/whisper-cli").resolve()


def test_locate_binary_missing(monkeypatch):
    monkeypatch.delenv(BIN_ENV, raising=False)
    monkeypatch.delenv(DIR_ENV, raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("engines.whisper_cpp.bootstrap._PKG_BIN_DIR", Path("/nonexistent/bin"))

    assert locate_binary() is None
    assert is_installed() is False


def test_resolve_model_direct_file(tmp_path):
    fake_model = tmp_path / "ggml-base.bin"
    fake_model.write_text("weights")

    assert resolve_model(str(fake_model)) == fake_model.resolve()


def test_resolve_model_in_user_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / ".cache" / "whisper"
    cache_dir.mkdir(parents=True)
    fake_model = cache_dir / "ggml-base.bin"
    fake_model.write_text("weights")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert resolve_model("base") == fake_model.resolve()


def test_is_available_verdicts(tmp_path, monkeypatch):
    # Neither binary nor model
    monkeypatch.setattr("engines.whisper_cpp.backend.locate_binary", lambda: None)
    monkeypatch.setattr("engines.whisper_cpp.backend.resolve_model", lambda: None)
    ok, msg = WhisperCppBackend.is_available()
    assert ok is False
    assert "binary not found" in msg

    # Binary found, model missing
    monkeypatch.setattr("engines.whisper_cpp.backend.locate_binary", lambda: tmp_path / "whisper-cli")
    ok, msg = WhisperCppBackend.is_available()
    assert ok is False
    assert "model not found" in msg

    # Both found
    monkeypatch.setattr("engines.whisper_cpp.backend.resolve_model", lambda: tmp_path / "ggml-base.bin")
    ok, msg = WhisperCppBackend.is_available()
    assert ok is True
    assert msg == "ok"


# ── GPU Compatibility & Routing Tests ─────────────────────────────────────────


def test_gpu_compat_and_routing():
    assert WhisperCppBackend.gpu_compat == ("vulkan", "cuda", "cpu")
    assert WhisperCppBackend.id == "whisper-cpp"

    # Vulkan-capable AMD host with Vulkan runtime note
    vulkan_caps = HostCaps(
        family="cpu",
        available_families=("cpu",),
        notes=(f"{VULKAN_MARKER}; Vulkan-native engines can use the GPU",),
    )
    routing = resolve_routing(WhisperCppBackend.gpu_compat, vulkan_caps)
    assert routing["effective_device"] == "vulkan"
    assert routing["routing_status"] == "accelerated"

    # Plain CPU host without Vulkan marker
    cpu_caps = HostCaps(family="cpu", available_families=("cpu",), notes=())
    routing_cpu = resolve_routing(WhisperCppBackend.gpu_compat, cpu_caps)
    assert routing_cpu["effective_device"] == "cpu"
    assert routing_cpu["routing_status"] == "cpu_only"


# ── Transcription & JSON Parsing Tests ────────────────────────────────────────


_SAMPLE_WHISPER_CPP_JSON = {
    "systeminfo": "AVX2 = 1 | VULKAN = 1",
    "model": {"type": "base"},
    "result": {"language": "en"},
    "transcription": [
        {
            "timestamps": {"from": "00:00:00,000", "to": "00:00:02,500"},
            "offsets": {"from": 0, "to": 2500},
            "text": "Hello world.",
            "tokens": [
                {
                    "text": " Hello",
                    "offsets": {"from": 0, "to": 1200},
                    "p": 0.95,
                },
                {
                    "text": " world.",
                    "offsets": {"from": 1200, "to": 2500},
                    "p": 0.98,
                },
            ],
        },
        {
            "timestamps": {"from": "00:00:02,500", "to": "00:00:04,800"},
            "offsets": {"from": 2500, "to": 4800},
            "text": "This is whisper cpp.",
            "tokens": [
                {
                    "text": " This",
                    "offsets": {"from": 2500, "to": 3100},
                    "p": 0.92,
                },
                {
                    "text": " is",
                    "offsets": {"from": 3100, "to": 3400},
                    "p": 0.96,
                },
                {
                    "text": " whisper",
                    "offsets": {"from": 3400, "to": 4100},
                    "p": 0.99,
                },
                {
                    "text": " cpp.",
                    "offsets": {"from": 4100, "to": 4800},
                    "p": 0.97,
                },
            ],
        },
    ],
}


def test_transcribe_success(tmp_path, monkeypatch):
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_text("RIFFdummyWAVE")
    fake_bin = tmp_path / "whisper-cli"
    fake_model = tmp_path / "ggml-base.bin"

    monkeypatch.setattr("engines.whisper_cpp.backend.locate_binary", lambda: fake_bin)
    monkeypatch.setattr("engines.whisper_cpp.backend.resolve_model", lambda: fake_model)
    monkeypatch.setattr("engines.whisper_cpp.backend._ensure_wav_16k", lambda path, tmp: Path(path))

    executed_cmds = []

    def _fake_run(cmd, **kw):
        executed_cmds.append(cmd)
        # Find -of prefix
        of_idx = cmd.index("-of")
        out_prefix = Path(cmd[of_idx + 1])
        out_json = out_prefix.parent / "out.json"
        out_json.write_text(json.dumps(_SAMPLE_WHISPER_CPP_JSON), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    backend = WhisperCppBackend()
    res = backend.transcribe(str(fake_audio), word_timestamps=True, language="en")

    assert res["text"] == "Hello world. This is whisper cpp."
    assert res["language"] == "en"
    assert len(res["chunks"]) == 2
    assert res["chunks"][0] == {"text": "Hello world.", "timestamp": (0.0, 2.5)}
    assert res["chunks"][1] == {"text": "This is whisper cpp.", "timestamp": (2.5, 4.8)}

    # Verify word timestamps
    assert len(res["segments"]) == 2
    words_0 = res["segments"][0]["words"]
    assert len(words_0) == 2
    assert words_0[0]["word"] == "Hello"
    assert words_0[0]["start"] == 0.0
    assert words_0[0]["end"] == 1.2

    # Verify command line arguments passed
    assert len(executed_cmds) == 1
    cmd = executed_cmds[0]
    assert "-m" in cmd
    assert "-l" in cmd
    assert "en" in cmd
    assert "--output-json-full" in cmd


def test_transcribe_device_and_backend_flags(tmp_path, monkeypatch):
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_text("RIFFdummyWAVE")
    fake_bin = tmp_path / "whisper-cli"
    fake_model = tmp_path / "ggml-base.bin"

    monkeypatch.setattr("engines.whisper_cpp.backend.locate_binary", lambda: fake_bin)
    monkeypatch.setattr("engines.whisper_cpp.backend.resolve_model", lambda: fake_model)
    monkeypatch.setattr("engines.whisper_cpp.backend._ensure_wav_16k", lambda path, tmp: Path(path))

    executed_cmds = []

    def _fake_run(cmd, **kw):
        executed_cmds.append(cmd)
        of_idx = cmd.index("-of")
        out_prefix = Path(cmd[of_idx + 1])
        (out_prefix.parent / "out.json").write_text(json.dumps(_SAMPLE_WHISPER_CPP_JSON))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    backend = WhisperCppBackend()

    # Test CPU backend flag
    monkeypatch.setenv(BACKEND_ENV, "cpu")
    backend.transcribe(str(fake_audio), word_timestamps=False)
    assert "-ng" in executed_cmds[0]

    # Test device index flag
    monkeypatch.delenv(BACKEND_ENV)
    monkeypatch.setenv(DEVICE_ENV, "1")
    backend.transcribe(str(fake_audio), task="translate")
    assert "-dev" in executed_cmds[1]
    assert "1" in executed_cmds[1]
    assert "--translate" in executed_cmds[1]


def test_transcribe_missing_audio_file():
    backend = WhisperCppBackend()
    with pytest.raises(FileNotFoundError):
        backend.transcribe("non_existent_audio_path.wav")


def test_transcribe_timeout_handling(tmp_path, monkeypatch):
    fake_audio = tmp_path / "test.wav"
    fake_audio.write_text("RIFFdummyWAVE")
    monkeypatch.setattr("engines.whisper_cpp.backend.locate_binary", lambda: tmp_path / "whisper-cli")
    monkeypatch.setattr("engines.whisper_cpp.backend.resolve_model", lambda: tmp_path / "ggml-base.bin")
    monkeypatch.setattr("engines.whisper_cpp.backend._ensure_wav_16k", lambda path, tmp: Path(path))

    def _fake_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="whisper-cli", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _fake_timeout)

    backend = WhisperCppBackend()
    with pytest.raises(ab.ASSRTimeoutError if hasattr(ab, "ASSRTimeoutError") else ab.ASRTimeoutError):
        backend.transcribe(str(fake_audio))


# ── ASR Auto-Detect Integration Tests ─────────────────────────────────────────


def test_autodetect_picks_whisper_cpp_on_vulkan_amd_host(monkeypatch):
    monkeypatch.setattr(ab, "_mps_available", lambda: False)
    monkeypatch.setattr(ab, "_cuda_reported_available", lambda: False)
    monkeypatch.setattr(ab, "_vulkan_preferred", lambda: True)

    # Stub probe_available: whisper-cpp is available, whisperx is available
    monkeypatch.setattr(
        ab,
        "_probe_available",
        lambda cls: cls.id in ("whisper-cpp", "whisperx", "faster-whisper"),
    )

    assert ab._auto_detect() == "whisper-cpp"


def test_autodetect_falls_back_when_whisper_cpp_unavailable(monkeypatch):
    monkeypatch.setattr(ab, "_mps_available", lambda: False)
    monkeypatch.setattr(ab, "_cuda_reported_available", lambda: False)
    monkeypatch.setattr(ab, "_vulkan_preferred", lambda: True)

    # whisper-cpp is not available, whisperx is available
    monkeypatch.setattr(
        ab,
        "_probe_available",
        lambda cls: cls.id in ("whisperx", "faster-whisper"),
    )

    assert ab._auto_detect() == "whisperx"


def test_autodetect_prefers_cuda_whisperx_on_nvidia_host(monkeypatch):
    monkeypatch.setattr(ab, "_mps_available", lambda: False)
    monkeypatch.setattr(ab, "_cuda_reported_available", lambda: True)
    monkeypatch.setattr(ab, "_rocm_torch", lambda: False)

    # Both whisperx and whisper-cpp are available
    monkeypatch.setattr(
        ab,
        "_probe_available",
        lambda cls: cls.id in ("whisper-cpp", "whisperx", "faster-whisper"),
    )

    assert ab._auto_detect() == "whisperx"
