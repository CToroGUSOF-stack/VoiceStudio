"""whisper.cpp ASR backend implementation (Phase 3).

Native C/C++ speech recognition via whisper.cpp. Provides GPU-accelerated
transcription on AMD GPUs via the Vulkan runtime without requiring CTranslate2
or a PyTorch wheel swap.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from engines.whisper_cpp.bootstrap import (
    BACKEND_ENV,
    DEVICE_ENV,
    locate_binary,
    resolve_model,
)
from services.asr_backend import ASRBackend, ASRTimeoutError, ASR_TRANSCRIBE_TIMEOUT_S

logger = logging.getLogger("omnivoice.whisper_cpp")


def _parse_timestamp(val: str | int | float) -> float:
    """Parse milliseconds int or 'hh:mm:ss,mmm' timestamp to float seconds."""
    if isinstance(val, (int, float)):
        return float(val) / 1000.0 if val > 500 else float(val)
    s = str(val).strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    try:
        return float(s)
    except ValueError:
        return 0.0


def _ensure_wav_16k(audio_path: str, temp_dir: Path) -> Path:
    """Ensure audio is 16 kHz mono WAV for whisper.cpp.

    Returns audio_path if already compliant, or a path to a converted temporary file.
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        if info.samplerate == 16000 and info.channels == 1 and info.format == "WAV":
            return Path(audio_path)
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import numpy as np
            n = int(round(len(data) * 16000 / sr))
            if n > 0:
                xp = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
                x = np.linspace(0.0, 1.0, num=n, endpoint=False)
                data = np.interp(x, xp, data).astype(np.float32)
        out_wav = temp_dir / "input_16k.wav"
        sf.write(str(out_wav), data, 16000, subtype="PCM_16")
        return out_wav
    except Exception:
        # Fallback to ffmpeg if soundfile fails
        from services.ffmpeg_utils import find_ffmpeg
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return Path(audio_path)
        out_wav = temp_dir / "input_16k.wav"
        cmd = [
            str(ffmpeg), "-y", "-nostdin", "-threads", "0",
            "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out_wav),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return out_wav


class WhisperCppBackend(ASRBackend):
    """ASR backend powered by whisper.cpp with native Vulkan acceleration."""

    id: str = "whisper-cpp"
    display_name: str = "whisper.cpp (Vulkan)"
    # Vulkan first, followed by CUDA and CPU.
    gpu_compat: tuple[str, ...] = ("vulkan", "cuda", "cpu")
    requires_full_audio_for_speaker_consistency: bool = False

    def supports_translation(self) -> bool:
        return True

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        binary = locate_binary()
        if not binary:
            return (
                False,
                "whisper.cpp binary not found. Set OMNIVOICE_WHISPER_CPP_BIN or "
                "place whisper-cli in backend/engines/whisper_cpp/bin/.",
            )
        model = resolve_model()
        if not model:
            return (
                False,
                "whisper.cpp model not found. Set OMNIVOICE_WHISPER_CPP_MODEL or "
                "download a ggml-*.bin model.",
            )
        return True, "ok"

    def transcribe(
        self,
        audio_path: str,
        *,
        word_timestamps: bool = True,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        task: str = "transcribe",
    ) -> dict:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        binary = locate_binary()
        if not binary:
            raise RuntimeError("whisper.cpp binary is not available")

        model_path = resolve_model()
        if not model_path:
            raise RuntimeError("whisper.cpp model file could not be resolved")

        with tempfile.TemporaryDirectory(prefix="vs_whisper_cpp_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            wav_path = _ensure_wav_16k(audio_path, tmp_dir)
            out_prefix = tmp_dir / "out"

            cmd = [
                str(binary),
                "-m", str(model_path),
                "-f", str(wav_path),
                "--output-json-full" if word_timestamps else "-oj",
                "-of", str(out_prefix),
                "-nt",  # suppress timestamps in stdout for cleaner execution
            ]

            if language and language.strip().lower() not in ("auto", ""):
                cmd.extend(["-l", language.strip()])

            if task == "translate":
                cmd.append("--translate")

            if initial_prompt:
                cmd.extend(["--prompt", initial_prompt])

            if temperature is not None:
                cmd.extend(["--temperature", str(temperature)])

            backend_override = os.environ.get(BACKEND_ENV, "").strip().lower()
            device_override = os.environ.get(DEVICE_ENV, "").strip()

            if backend_override == "cpu":
                cmd.append("-ng")
            elif device_override:
                cmd.extend(["-dev", device_override])

            logger.info("Executing whisper.cpp: %s", " ".join(cmd[:6]))

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=ASR_TRANSCRIBE_TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ASRTimeoutError(
                    f"whisper.cpp transcription timed out after {ASR_TRANSCRIBE_TIMEOUT_S}s"
                ) from exc

            out_json = tmp_dir / "out.json"
            if not out_json.is_file():
                # If --output-json-full failed on older binary, retry with -oj
                if word_timestamps:
                    cmd_fallback = [
                        c if c != "--output-json-full" else "-oj" for c in cmd
                    ]
                    proc = subprocess.run(
                        cmd_fallback,
                        capture_output=True,
                        text=True,
                        timeout=ASR_TRANSCRIBE_TIMEOUT_S,
                        check=False,
                    )
                if not out_json.is_file():
                    err_msg = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
                    raise RuntimeError(f"whisper.cpp failed to generate transcript: {err_msg}")

            try:
                data = json.loads(out_json.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                raise RuntimeError(f"Failed to parse whisper.cpp JSON output: {exc}") from exc

        # Format output dictionary to match VoiceStudio ASR contract
        transcription = data.get("transcription") or []
        segments = []
        chunks = []

        for seg in transcription:
            seg_text = seg.get("text", "")
            offsets = seg.get("offsets", {})
            timestamps = seg.get("timestamps", {})
            start = (
                offsets.get("from") / 1000.0
                if "from" in offsets
                else _parse_timestamp(timestamps.get("from", 0.0))
            )
            end = (
                offsets.get("to") / 1000.0
                if "to" in offsets
                else _parse_timestamp(timestamps.get("to", 0.0))
            )

            words = []
            if word_timestamps:
                for tok in seg.get("tokens", []):
                    tok_text = tok.get("text", "")
                    if tok_text.startswith("[_") and tok_text.endswith("]"):
                        continue
                    tok_offsets = tok.get("offsets", {})
                    tok_timestamps = tok.get("timestamps", {})
                    tok_start = (
                        tok_offsets.get("from") / 1000.0
                        if "from" in tok_offsets
                        else _parse_timestamp(tok_timestamps.get("from", start))
                    )
                    tok_end = (
                        tok_offsets.get("to") / 1000.0
                        if "to" in tok_offsets
                        else _parse_timestamp(tok_timestamps.get("to", end))
                    )
                    words.append({
                        "word": tok_text.strip(),
                        "start": tok_start,
                        "end": tok_end,
                        "probability": float(tok.get("p", 1.0)),
                    })

            chunks.append({"text": seg_text, "timestamp": (start, end)})
            segments.append({
                "text": seg_text,
                "start": start,
                "end": end,
                "words": words,
            })

        full_text = " ".join(c["text"].strip() for c in chunks if c["text"].strip())
        detected_lang = data.get("result", {}).get("language") or "en"

        return {
            "text": full_text,
            "chunks": chunks,
            "segments": segments,
            "language": detected_lang,
        }
