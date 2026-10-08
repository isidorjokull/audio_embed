"""Reading audio files. Everything here is read-only against the library."""

import subprocess
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".caf"}

# Windows whose peak is below this (about -80 dBFS) are skipped as silence.
SILENCE_PEAK = 1e-4
# A leftover tail shorter than this is not worth its own window.
MIN_TAIL_S = 1.0


def is_local(path) -> bool:
    """Whether a file's contents are on this machine. A cloud placeholder is an empty file."""
    try:
        return Path(path).stat().st_size > 0
    except OSError:
        return False


def find_audio(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file() and not p.name.startswith("._")
    )


def channel_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode or not out.stdout.strip():
        raise RuntimeError(f"ffprobe failed: {out.stderr.strip()[:200]}")
    return int(out.stdout.strip().split(",")[0])


def decode(path: Path, sr: int) -> np.ndarray:
    """Decode to mono float32 at `sr`.

    Channels are averaged here rather than by ffmpeg, which refuses to downmix
    layouts it has no matrix for (ambisonic and other many-channel files).
    """
    channels = channel_count(path)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-ar", str(sr), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    frame_bytes = 4 * channels
    block = frame_bytes * sr * 10
    parts = []
    pending = b""
    while chunk := proc.stdout.read(block):
        pending += chunk
        usable = len(pending) - len(pending) % frame_bytes
        frames = np.frombuffer(pending[:usable], dtype="<f4").reshape(-1, channels)
        parts.append(frames.mean(axis=1, dtype=np.float32))
        pending = pending[usable:]
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed: {err.strip()[:200]}")
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def window_spans(n_samples: int, sr: int, window_s: float) -> list[tuple[int, int]]:
    """Split a file into (start, end) sample spans of at most `window_s`.

    A file shorter than one window is a single span. Otherwise spans tile the
    file, and a leftover tail is covered by one last full-length window pulled
    back to end at the file's end, so every span the model sees is full length.
    """
    size = int(window_s * sr)
    if n_samples <= 0:
        return []
    if n_samples <= size:
        return [(0, n_samples)]
    spans = [(s, s + size) for s in range(0, n_samples - size + 1, size)]
    tail = n_samples - spans[-1][1]
    if tail >= MIN_TAIL_S * sr:
        spans.append((n_samples - size, n_samples))
    return spans


def is_silent(samples: np.ndarray) -> bool:
    return samples.size == 0 or float(np.abs(samples).max()) < SILENCE_PEAK


OUTLINE_BUCKETS = 800


def outline(samples: np.ndarray, buckets: int = OUTLINE_BUCKETS) -> np.ndarray:
    """A waveform outline: the peak level in each of `buckets` slices, as bytes scaled so the loudest is 255.

    Takes samples that are already decoded, so indexing gets the outline for free.
    """
    samples = np.abs(samples)
    if samples.size == 0:
        return np.zeros(0, dtype=np.uint8)
    buckets = min(buckets, samples.size)
    edges = np.linspace(0, samples.size, buckets + 1, dtype=np.int64)
    levels = np.maximum.reduceat(samples, edges[:-1])
    top = float(levels.max())
    if top <= 0:
        return np.zeros(buckets, dtype=np.uint8)
    return np.round(levels / top * 255).astype(np.uint8)


def peaks(path: Path, buckets: int = OUTLINE_BUCKETS) -> list[float]:
    """The outline of a file as levels from 0 to 1, decoding it at a low rate for speed."""
    return (outline(decode(path, 8000), buckets) / 255).round(3).tolist()


def write_preview(path: Path, out: Path) -> None:
    """Transcode to a 48 kHz AAC file any browser can play and seek in: mono stays mono, the rest is stereo."""
    channels = channel_count(path)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0"]
    if channels == 1:
        # A lone channel is sometimes labelled "front left" rather than mono, which the encoder refuses.
        cmd += ["-af", "pan=mono|c0=c0"]
    elif channels > 2:
        # ffmpeg has no downmix for arbitrary layouts, so average every channel into both sides.
        mix = "+".join(f"{1 / channels:.4f}*c{c}" for c in range(channels))
        cmd += ["-af", f"pan=stereo|c0={mix}|c1={mix}"]
    cmd += ["-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            "-f", "mp4", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:200]}")
