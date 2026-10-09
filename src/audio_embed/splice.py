"""Arithmetic on the samples of a clip, with no model in it.

Stable Audio 3 can make one stretch of a clip anew while it hears the rest. It works
in steps of 4096 samples and keeps the rest exactly only in its own compressed form,
so this tool takes just the new stretch from what comes back and joins it into the
clip it sent. Everything here works on 44.1 kHz 16-bit stereo, the WAV that
`generate.cut` writes and Stable Audio 3 reads.
"""

import math
import wave
from pathlib import Path

import numpy as np

SR = 44100
# One step of Stable Audio 3's compressed form, in samples.
STEP = 4096
# How long the old and the new audio overlap at each edge of a part: 30 ms.
FADE = round(0.03 * SR)


def read(path: Path) -> np.ndarray:
    """The samples of a WAV as (n, 2) 16-bit integers."""
    with wave.open(str(path), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (SR, 2, 2):
            raise ValueError(f"{path} is not a 44.1 kHz 16-bit stereo WAV.")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").reshape(-1, 2).copy()


def write(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())


def snap(seconds: float) -> int:
    """The sample nearest a time that lies on a step."""
    return int(round(seconds * SR / STEP)) * STEP


def clock(samples: int) -> str:
    """A count of samples as the seconds Stable Audio 3 is told.

    It rounds seconds up to a whole step, so the time is written a little short
    (one to two millionths of a second): a whole number of steps is then never
    read as one step more, and the length of the file it writes is still exact.
    """
    return f"{max(0, math.floor(samples / SR * 1e6) - 1) / 1e6:.6f}"


def whole_steps(samples: np.ndarray) -> np.ndarray:
    """A clip without the end that does not fill a step."""
    return samples[: len(samples) // STEP * STEP]


def turn(samples: np.ndarray, by: int) -> np.ndarray:
    """A clip started `by` samples in, with what came before moved to the end, so its end and start meet inside it."""
    return np.roll(samples, -by, axis=0)


def turn_back(samples: np.ndarray, by: int) -> np.ndarray:
    return np.roll(samples, by, axis=0)


def join(original: np.ndarray, made: np.ndarray, start: int, end: int, fade: int = FADE) -> np.ndarray:
    """`original` with samples `start` to `end` taken from `made`, crossfaded just outside that part.

    An `end` past the end of `original` makes the clip that much longer.
    """
    if not 0 <= start < end:
        raise ValueError("The part to join has no length.")
    if len(made) < end:
        raise ValueError("Stable Audio 3 returned a shorter clip than was asked for.")
    kept = len(original)
    out = np.zeros((max(kept, end), 2), dtype=np.float64)
    out[:kept] = original
    out[start:end] = made[start:end]
    before = min(fade, start)
    if before:
        ramp = (np.arange(1, before + 1) / (before + 1))[:, None]
        out[start - before:start] = original[start - before:start] * (1 - ramp) + made[start - before:start] * ramp
    after = max(0, min(fade, kept - end, len(made) - end))
    if after:
        ramp = (np.arange(1, after + 1) / (after + 1))[:, None]
        out[end:end + after] = made[end:end + after] * (1 - ramp) + original[end:end + after] * ramp
    return np.clip(np.rint(out), -32768, 32767).astype("<i2")
