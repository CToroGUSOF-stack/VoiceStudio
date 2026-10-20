"""Local audio normalization with anti-alias filtering."""
import subprocess
from pathlib import Path


def _ensure_wav_16k(audio_path: str, temp_dir: Path) -> Path:
    """Ensure audio is 16 kHz mono WAV for whisper.cpp.

    Returns audio_path if already compliant, or a path to a converted temporary file.
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        if info.samplerate == 16000 and info.channels == 1 and info.format == "WAV" and info.subtype == "PCM_16":
            return Path(audio_path)
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            import torch
            import torchaudio
            data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
        out_wav = temp_dir / "input_16k.wav"
        sf.write(str(out_wav), data, 16000, subtype="PCM_16")
        return out_wav
    except (ImportError, OSError, RuntimeError, ValueError):
        # Fallback to ffmpeg if soundfile fails
        from services.ffmpeg_utils import find_ffmpeg
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("Audio conversion requires ffmpeg or torchaudio")
        out_wav = temp_dir / "input_16k.wav"
        cmd = [
            str(ffmpeg), "-y", "-nostdin", "-threads", "0",
            "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out_wav),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        return out_wav
