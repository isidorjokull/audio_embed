# Redo a part, loop, make longer: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the generate view regenerate one part of a sample, make a sample loop, or continue it past its end, from a library file or from a clip that has not been kept.

**Architecture:** All three are one Stable Audio 3 call (`--init-audio` with `--inpaint-range`) with the regenerated stretch placed differently. A new `splice.py` does the sample arithmetic with no model in it; `generate.py` chooses the passage and the stretch for each kind, runs the model, and joins only the new stretch back into the passage it sent, so everything else in a clip is the cut passage sample for sample. The server and page pass the new choices through.

**Tech Stack:** Python 3 with NumPy and the standard `wave` module; ffmpeg for cutting; Starlette routes; one HTML file with plain JavaScript and a canvas; pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-inpainting-design.md`

## Listening results (Task 2)

**Verdict of 2026-10-09: part, loop and longer are all built.** They pass on sustained sounds (a drone, an ambience). Longer does not work on sparse events (doors in a quiet room): the new material is garble. Part and loop were not really tested on events, because the stretches redone in the doors file were room tone.

First probe: the first 20 seconds of three files, no prompt, seed 1, medium and small sfx models.

- **Longer, drone, medium:** the new 10 seconds change in vibe from the start. The user likes the result, but it is a different sound.
- **Longer, ambience, medium:** the texture changes suddenly at the end.
- **Longer, ambience, small:** closer to the original than medium, with more artifacts.
- **Third file (kettle, a raw geophone recording of knocking and noise):** not a useful example. Replace it before judging anything on it.
- **Part and loop:** "sounded good" on the drone and the ambience, both models. Not yet heard on a file with clear events.

Second probe, medium model only: Longer on the drone and the ambience with three seeds, a prompt, 60 seconds of the file, and both; part, loop and Longer on `Foley/Doors/Hurðir inná litlasviði.wav` in place of the kettle. Each file is 20 seconds of the original and 10 seconds of new material.

- **Longer, drone:** all six are fine, "a bit thinner than the original" (they measure about 3 dB quieter). Seed 3 is the best of the three without a prompt. With a prompt is the most realistic continuation. The two made from 60 seconds are liked as well.
- **Longer, ambience:** all six are "believable and usable". The three seeds each go somewhere different (seed 1 evolves somewhere new, seed 2 turns softer, seed 3 is more subdued), and the user likes that as an evolution. With a prompt there is less going on, and it works as a way to fade the file out. The two made from 60 seconds are the best: "complex texture and usable".
- **Part and loop, doors:** silent at the edit and at the loop point. The original is room tone in both places (about -55 dB), so that is what was there, and the test says nothing about an edit across an event.
- **Longer, doors:** garble without a prompt; less so from 60 seconds with a prompt, still unusable.

What this means for the build: nothing in the plan changes. Sending up to 60 seconds of the file (already the plan) gave the best continuations, and seeds differ enough that Count matters.

## Global Constraints

- The audio library is read-only and is read only through ffmpeg/ffprobe (`generate.cut`). All arithmetic happens on WAV files in a scratch folder.
- Never write test data into `data/`. Tests use `tmp_path`. Hand tests use a scratch index, a scratch kept folder and another port (`--db SCRATCH/data/index.db serve --port 8766 --no-open`).
- The server takes a clip **id** from the page (16 hex digits, checked by `Generator.clip`) and never a path.
- `web/index.html` stays one file with no build step and no external assets.
- Tests load no models and must stay fast: `uv run pytest -q`.
- Stable Audio 3 is given at most 60 seconds (`LONGEST_S`), 120 when a length is asked for (`LONGEST_ASKED_S`).
- Join lengths: Short 1 s, Medium 2 s, Long 4 s, half at each end; at most half the clip. A loop needs at least 2 seconds.
- Add lengths on the page: 10, 30, 60 seconds. Passage and new material together are at most 120 seconds.
- A redone part is at least 0.2 seconds and leaves at least 1 second of the passage untouched.
- One step of Stable Audio 3 is 4096 samples at 44.1 kHz. Crossfades are 30 ms (1323 samples).
- Clips are 44.1 kHz, 16-bit, stereo WAV, as now.
- **Do not run Stable Audio 3, or anything else that makes the Mac loud, without the user saying so.** Task 2 is the only task that runs it.
- Commit only on a branch, and end each commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A part marked right up to the end of a short one-shot.** Expected: the clip keeps its full length and the tail is made anew, with no sliver of old audio left after it. Test in Task 5 (`lay_out`).
2. **Stable Audio 3 returns a clip shorter than was asked for.** Expected: that clip fails with a sentence saying so, the others in the run are still made. Test in Task 6.
3. **A Long join asked of a 2-second sample.** Expected: the join shrinks to half the clip instead of failing or overlapping itself. Test in Task 4 (`plan`) and Task 5.
4. **Numbers from the page that are missing, not numbers, NaN, or a marker past the end of the file.** Expected: a 400 with a sentence, or the marker pulled back to the end. Tests in Tasks 3 and 4.
5. **Working from a clip that has since been cleared from the cache.** Expected: the request is refused with "That clip is no longer in the cache", nothing crashes. Test in Task 7.

Also pinned in Task 7: a story written before this change (no `make`) is still read and described as a variation.

## File Structure

| File | Responsibility |
|---|---|
| `src/audio_embed/splice.py` (new) | Sample arithmetic: read and write the WAV, step positions, turning a clip, joining a new part in. No model, no ffmpeg. |
| `src/audio_embed/generate.py` | `Ask` fields and checks; `plan` (which passage, which part); `lay_out`/`finish` (what the model is given and what is kept); `command`; the run; stories and `describe`; runs started from a clip. |
| `src/audio_embed/server.py` | Pass `make`, `span`, `marker`, `join`, `add`, `clip` through; describe a clip as the sample. |
| `src/audio_embed/web/index.html` | Make row, Join, Add, the part and marker on the waveform, Work from this, repeat for loops. |
| `tests/test_splice.py` (new) | Tests of `splice.py`. |
| `tests/test_generate.py` | Tests of everything in `generate.py`. |

## Before starting

- [ ] The working tree on `main` has uncommitted changes to `README.md`, `src/audio_embed/labels.py`, `src/audio_embed/web/index.html` and `tests/test_core.py` from earlier work. Ask the user whether to commit them first. Line numbers for `index.html` in this plan are from the file with those changes in it.
- [ ] Make a branch: `git switch -c inpainting`.
- [ ] Run `uv run pytest -q` and note that everything passes before any change.

---

### Task 1: `splice.py`, the sample arithmetic

**Files:**
- Create: `src/audio_embed/splice.py`
- Test: `tests/test_splice.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `splice.SR = 44100`, `splice.STEP = 4096`, `splice.FADE = 1323`
  - `splice.read(path) -> np.ndarray` of shape `(n, 2)`, dtype `int16`; raises `ValueError` for any other format
  - `splice.write(path, samples) -> None`
  - `splice.snap(seconds: float) -> int`: the sample on a step nearest a time
  - `splice.clock(samples: int) -> str`: seconds with six decimals, written just short of the true time
  - `splice.whole_steps(samples) -> np.ndarray`
  - `splice.turn(samples, by: int) -> np.ndarray`, `splice.turn_back(samples, by: int) -> np.ndarray`
  - `splice.join(original, made, start: int, end: int, fade: int = FADE) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_splice.py`:

```python
import math
import wave

import numpy as np
import pytest

from audio_embed import splice


def clip(n, value):
    return np.full((n, 2), value, dtype="<i2")


def test_a_wav_is_read_back_as_it_was_written(tmp_path):
    samples = (np.arange(2000, dtype="<i2").reshape(-1, 2) - 500).copy()
    splice.write(tmp_path / "a.wav", samples)
    assert np.array_equal(splice.read(tmp_path / "a.wav"), samples)
    with wave.open(str(tmp_path / "a.wav")) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (44100, 2, 2)


def test_a_wav_of_another_shape_is_refused(tmp_path):
    with wave.open(str(tmp_path / "mono.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b"\0\0" * 10)
    with pytest.raises(ValueError, match="44.1 kHz 16-bit stereo"):
        splice.read(tmp_path / "mono.wav")


def test_snap_moves_a_time_to_the_nearest_step():
    assert splice.snap(0.0) == 0
    assert splice.snap(0.1) == 4096        # 1.08 steps
    assert splice.snap(1.0) == 11 * 4096   # 10.77 steps
    assert splice.snap(0.04) == 0          # 0.43 steps


def test_clock_is_read_by_stable_audio_3_as_the_same_step_and_length():
    assert splice.clock(0) == "0.000000"
    assert splice.clock(45056) == "1.021677"
    for steps in (1, 2, 21, 441, 646, 882, 1292):   # up to two minutes; 441 steps is 40.96 s exactly
        n = steps * splice.STEP
        seconds = float(splice.clock(n))
        assert math.ceil(seconds * splice.SR / splice.STEP) == steps   # how many steps it makes
        assert round(seconds * splice.SR / splice.STEP) == steps       # where a part's edge lands
        assert int(round(seconds * splice.SR)) == n                    # how long its file is


def test_whole_steps_drops_what_does_not_fill_a_step():
    assert len(splice.whole_steps(clip(4096 * 3 + 100, 1))) == 4096 * 3
    assert len(splice.whole_steps(clip(4096 * 3, 1))) == 4096 * 3


def test_a_turn_puts_a_sample_first_and_turning_back_undoes_it():
    samples = np.arange(20, dtype="<i2").reshape(-1, 2).copy()
    turned = splice.turn(samples, 3)
    assert turned[0].tolist() == samples[3].tolist()
    assert turned[-1].tolist() == samples[2].tolist()
    assert np.array_equal(splice.turn_back(turned, 3), samples)


def test_only_the_part_and_its_fades_differ_from_the_original():
    original, made = clip(40000, 100), clip(40000, 9000)
    out = splice.join(original, made, 8192, 16384)
    assert out.dtype == np.dtype("<i2") and len(out) == 40000
    assert np.array_equal(out[: 8192 - splice.FADE], original[: 8192 - splice.FADE])
    assert np.array_equal(out[16384 + splice.FADE:], original[16384 + splice.FADE:])
    assert np.array_equal(out[8192:16384], made[8192:16384])
    before = out[8192 - splice.FADE: 8192, 0]
    assert 100 < before[0] < before[-1] < 9000 and np.all(np.diff(before) >= 0)
    after = out[16384: 16384 + splice.FADE, 0]
    assert 9000 > after[0] > after[-1] > 100 and np.all(np.diff(after) <= 0)


def test_a_part_at_either_end_has_no_fade_there():
    original, made = clip(20000, 100), clip(20000, 9000)
    assert np.array_equal(splice.join(original, made, 0, 4096)[:4096], made[:4096])
    assert np.array_equal(splice.join(original, made, 16000, 20000)[16000:], made[16000:])


def test_a_part_past_the_end_makes_the_clip_longer():
    original, made = clip(8192, 100), clip(20480, 9000)
    out = splice.join(original, made, 8192, 20480)
    assert len(out) == 20480
    assert np.array_equal(out[: 8192 - splice.FADE], original[: 8192 - splice.FADE])
    assert np.array_equal(out[8192:], made[8192:])


def test_a_made_clip_that_is_too_short_is_refused():
    with pytest.raises(ValueError, match="shorter clip"):
        splice.join(clip(20000, 1), clip(10000, 2), 8192, 16384)
    with pytest.raises(ValueError, match="no length"):
        splice.join(clip(20000, 1), clip(20000, 2), 8192, 8192)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_splice.py -q`
Expected: an import error, `cannot import name 'splice'`.

- [ ] **Step 3: Write `src/audio_embed/splice.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_splice.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/audio_embed/splice.py tests/test_splice.py
git commit -m "Add splice: join a newly made part into the clip it was made for"
```

---

### Task 2: Listen before building (needs the user's go-ahead; runs Stable Audio 3)

The spec says none of this has been heard. This task makes one of each kind by hand so the user can decide which kinds are worth building. **Do not start it until the user says the Mac may make noise.** Nothing is written to `data/` or the library.

**Files:**
- Create (in a scratch folder, not in the repo): `SCRATCH/probe.py`

**Interfaces:**
- Consumes: `splice.*` from Task 1; `generate.cut(source, start, seconds, out)` (exists).
- Produces: a verdict from the user for each of part, loop and longer. Tasks 3 to 11 are built for the kinds that pass; a kind that fails is left out of `MAKES` in Task 3 and of `MAKES` on the page in Task 9.

- [ ] **Step 1: Write the probe**

Save as `SCRATCH/probe.py` (replace `SCRATCH` with a new empty folder outside the repo):

```python
"""Make one redone part, one loop and one continuation of a file, for listening. Throwaway."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from audio_embed import generate, splice

SA3 = Path("/Users/isidor/git/stable-audio-3/optimized/mlx")
source, out, dit, decoder = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)


def sa3(reference, start, end, total, name, prompt=""):
    made = out / f"raw-{name}.wav"
    subprocess.run([
        str(SA3 / ".venv/bin/python"), str(SA3 / "scripts/sa3_mlx.py"), "--prompt", prompt, "--dit", dit, "--decoder", decoder,
        "--init-audio", str(reference), "--inpaint-range", f"{splice.clock(start)},{splice.clock(end)}",
        "--seconds", splice.clock(total), "--seed", "1", "--out", str(made),
    ], cwd=SA3, check=True)
    return splice.read(made)


reference = out / "reference.wav"
generate.cut(source, 0.0, 20.0, reference)
given = splice.whole_steps(splice.read(reference))
n, steps = len(given), len(given) // splice.STEP
splice.write(reference, given)

# A part: two seconds from a third of the way in.
a, b = splice.snap(n / 3 / splice.SR), splice.snap(n / 3 / splice.SR + 2.0)
splice.write(out / f"{dit}-part.wav", splice.join(given, sa3(reference, a, b, n, "part"), a, b))

# A loop with a two-second join, written three times over so the join is heard twice.
by = steps // 2 * splice.STEP
turned = out / "turned.wav"
splice.write(turned, splice.turn(given, by))
start = n - by - 11 * splice.STEP
end = start + 22 * splice.STEP
loop = splice.turn_back(splice.join(splice.turn(given, by), sa3(turned, start, end, n, "loop"), start, end), by)
splice.write(out / f"{dit}-loop-three-times.wav", np.concatenate([loop, loop, loop]))

# Longer: ten seconds after the end.
total = n + splice.snap(10.0)
splice.write(out / f"{dit}-longer.wav", splice.join(given, sa3(reference, n, total, total, "longer"), n, total))
print("Listen to the files in", out)
```

- [ ] **Step 2: Run it on three files, with the medium model and the small sfx model**

Pick three library files with the user (a drone or pad, an ambience, something with clear events). For each:

```bash
uv run python SCRATCH/probe.py "/path/to/library file.wav" SCRATCH/out-1 medium same-l
uv run python SCRATCH/probe.py "/path/to/library file.wav" SCRATCH/out-1 sm-sfx same-s
```

Expected: each run prints `Listen to the files in ...` and leaves `<model>-part.wav`, `<model>-loop-three-times.wav` and `<model>-longer.wav`. A run reads the library file only through ffmpeg.

- [ ] **Step 3: The user listens and decides**

Ask the user, for each kind and model: are the edges of the part audible; does the loop repeat without a seam (the join is at 20 s and 40 s of the three-times file); does the continuation follow on. Record the answers at the top of this plan. If the 30 ms fade is audible, try `FADE = round(0.1 * SR)` in `splice.py` and run again before deciding. Remove the scratch folder when done.

---

### Task 3: What the panel asks for

**Files:**
- Modify: `src/audio_embed/generate.py` (constants near line 42, `Ask` at 57, `ask_from` at 70)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `generate.MAKES = ("variations", "part", "loop", "longer")`
  - `generate.JOINS = {"short": 1.0, "medium": 2.0, "long": 4.0}`
  - `generate.SHORTEST_PART_S = 0.2`, `generate.MIN_KEPT_S = 1.0`, `generate.MIN_LOOP_S = 2.0`
  - `Ask` gains `make: str = "variations"`, `span: tuple[float, float] | None = None`, `marker: float | None = None`, `join: str = "medium"`, `add: float = 30.0`
  - `ask_from(body)` reads `make`, `span` (`[start, end]` in seconds in the sample), `marker` (seconds in the sample, or null for its end), `join`, `add`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_generate.py`:

```python
def test_a_request_says_what_to_make_and_where():
    ask = generate.ask_from({"make": "part", "span": [3, "4.5"], "prompt": "bells"})
    assert (ask.make, ask.span, ask.prompt) == ("part", (3.0, 4.5), "bells")
    assert generate.ask_from({"make": "loop", "join": "long"}).join == "long"
    longer = generate.ask_from({"make": "longer", "add": 10, "marker": 7.5})
    assert (longer.add, longer.marker) == (10.0, 7.5)
    assert generate.ask_from({"make": "longer"}).marker is None   # the end of the sample
    plain = generate.ask_from({})
    assert (plain.make, plain.span, plain.marker, plain.join, plain.add) == ("variations", None, None, "medium", 30.0)


@pytest.mark.parametrize("body, says", [
    ({"make": "remix"}, "What to make"),
    ({"make": "loop", "join": "huge"}, "The join"),
    ({"make": "part"}, "Mark the part"),
    ({"make": "part", "span": [5, 5]}, "start before it ends"),
    ({"make": "part", "span": [5, 4]}, "start before it ends"),
    ({"make": "part", "span": [-1, 4]}, "start before it ends"),
    ({"make": "part", "span": [1, 1.1]}, "at least 0.2 seconds"),
    ({"make": "part", "span": ["x", 4]}, "must be numbers"),
    ({"make": "part", "span": [4]}, "must be numbers"),
    ({"make": "part", "span": [float("nan"), 4]}, "start before it ends"),
    ({"make": "longer", "add": 0}, "Between 1 and 119"),
    ({"make": "longer", "add": 500}, "Between 1 and 119"),
    ({"make": "longer", "add": float("nan")}, "Between 1 and 119"),
    ({"make": "longer", "marker": 0}, "after the start"),
    ({"make": "longer", "marker": float("nan")}, "after the start"),
])
def test_a_request_for_a_part_a_loop_or_more_is_checked(body, says):
    with pytest.raises(ValueError, match=says):
        generate.ask_from(body)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_generate.py -q -k "says_what_to_make or is_checked"`
Expected: FAIL, `Ask` has no `make`.

- [ ] **Step 3: Add the constants and fields**

In `src/audio_embed/generate.py`, after the `DISTANCES` line:

```python
# What a run makes: whole new clips, or one stretch of the sample made anew (see splice.py).
MAKES = ("variations", "part", "loop", "longer")
# How many seconds a loop's join replaces, half of it at each end of the clip.
JOINS = {"short": 1.0, "medium": 2.0, "long": 4.0}
# The shortest part that can be redone, and how much of a passage a part must leave alone.
SHORTEST_PART_S = 0.2
MIN_KEPT_S = 1.0
MIN_LOOP_S = 2.0
```

Add to the end of `Ask`:

```python
    make: str = "variations"
    span: tuple[float, float] | None = None  # the part to redo, in seconds in the sample
    marker: float | None = None  # where new sound takes over, for longer; None: the end of the sample
    join: str = "medium"
    add: float = 30.0  # seconds of new sound, for longer
```

In `ask_from`, replace the last line (`return Ask(text("prompt"), avoid, distance, strength, count, seconds, model)`) with:

```python
    make, join = body.get("make", "variations"), body.get("join", "medium")
    if make not in MAKES:
        raise ValueError(f"What to make must be one of: {', '.join(MAKES)}.")
    if join not in JOINS:
        raise ValueError(f"The join must be one of: {', '.join(JOINS)}.")
    try:
        add = float(body.get("add", 30.0))
        span = None if body.get("span") is None else (float(body["span"][0]), float(body["span"][1]))
        marker = None if body.get("marker") in (None, "") else float(body["marker"])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ValueError("The part, the marker and the length to add must be numbers.") from None
    if make == "part":
        if span is None:
            raise ValueError("Mark the part to redo on the waveform first.")
        if not 0 <= span[0] < span[1]:
            raise ValueError("The part must start before it ends.")
        if span[1] - span[0] < SHORTEST_PART_S:
            raise ValueError(f"A part must be at least {SHORTEST_PART_S:g} seconds long.")
    if make == "longer" and not 1 <= add <= LONGEST_ASKED_S - MIN_KEPT_S:
        raise ValueError(f"Between 1 and {LONGEST_ASKED_S - MIN_KEPT_S:g} seconds can be added.")
    if marker is not None and not marker > 0:
        raise ValueError("The marker must be after the start of the sample.")
    return Ask(text("prompt"), avoid, distance, strength, count, seconds, model, make, span, marker, join, add)
```

The comparisons are written so that NaN fails them (`not 0 <= nan` is true, `not nan > 0` is true).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_generate.py -q`
Expected: all pass, including the tests that were there before.

- [ ] **Step 5: Commit**

```bash
git add src/audio_embed/generate.py tests/test_generate.py
git commit -m "Let a generate request ask for a part, a loop or more of a sample"
```

---

### Task 4: Which passage, and which part of it

**Files:**
- Modify: `src/audio_embed/generate.py` (after `passage`, near line 108)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `Ask` and the constants from Task 3; `passage(duration, matched_at, asked)` (exists).
- Produces:
  - `generate.Plan(start_s, seconds, whole, part=None, join_s=0.0, out_s=0.0)`, a frozen dataclass. `start_s` and `seconds` are the passage cut from the sample; `part` is `(start, end)` in seconds from the start of the passage (for a part and for longer); `join_s` is the join of a loop; `out_s` is how long the clip will be.
  - `generate.plan(duration: float, matched_at: float, ask: Ask) -> Plan`; raises `ValueError` with a sentence for the page.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_passage_for_a_part_is_placed_around_it():
    short = generate.plan(10.0, 0.0, Ask(make="part", span=(3.0, 5.0)))
    assert (short.start_s, short.seconds, short.whole, short.part, short.out_s) == (0.0, 10.0, True, (3.0, 5.0), 10.0)
    long = generate.plan(600.0, 0.0, Ask(make="part", span=(300.0, 310.0)))
    assert (long.start_s, long.seconds, long.whole) == (275.0, 60.0, False)   # as much before as after
    assert long.part == (25.0, 35.0)
    at_the_end = generate.plan(600.0, 0.0, Ask(make="part", span=(595.0, 600.0)))
    assert (at_the_end.start_s, at_the_end.part) == (540.0, (55.0, 60.0))     # pulled back to fit
    past = generate.plan(10.0, 0.0, Ask(make="part", span=(8.0, 99.0)))
    assert past.part == (8.0, 10.0)                                            # never past the end


@pytest.mark.parametrize("duration, span, says", [
    (10.0, (0.0, 9.5), "Mark a shorter part"),        # under a second would stay
    (600.0, (100.0, 160.0), "Mark a shorter part"),   # as long as the passage itself
    (10.0, (9.9, 12.0), "past the end"),
    (10.0, (20.0, 22.0), "past the end"),
])
def test_a_part_that_leaves_nothing_to_match_is_refused(duration, span, says):
    with pytest.raises(ValueError, match=says):
        generate.plan(duration, 0.0, Ask(make="part", span=span))


def test_a_loop_is_the_passage_variations_would_use_with_a_join_that_fits():
    loop = generate.plan(600.0, 200.0, Ask(make="loop", join="long", seconds=20.0))
    assert (loop.start_s, loop.seconds, loop.join_s, loop.part, loop.out_s) == (200.0, 20.0, 4.0, None, 20.0)
    tight = generate.plan(2.0, 0.0, Ask(make="loop", join="long"))
    assert tight.join_s == 1.0   # never more than half the clip
    with pytest.raises(ValueError, match="at least 2 seconds"):
        generate.plan(1.5, 0.0, Ask(make="loop"))


def test_longer_follows_on_from_the_marker_or_the_end():
    from_the_end = generate.plan(8.0, 0.0, Ask(make="longer", add=10.0))
    assert (from_the_end.start_s, from_the_end.seconds, from_the_end.whole) == (0.0, 8.0, True)
    assert (from_the_end.part, from_the_end.out_s) == ((8.0, 18.0), 18.0)
    pulled_back = generate.plan(8.0, 0.0, Ask(make="longer", add=10.0, marker=6.0))
    assert (pulled_back.seconds, pulled_back.whole, pulled_back.part) == (6.0, False, (6.0, 16.0))
    long = generate.plan(600.0, 0.0, Ask(make="longer", add=30.0))
    assert (long.start_s, long.seconds, long.out_s) == (540.0, 60.0, 90.0)     # the last minute of it
    most = generate.plan(600.0, 0.0, Ask(make="longer", add=100.0))
    assert (most.seconds, most.out_s) == (20.0, 120.0)                         # never more than two minutes in all
    beyond = generate.plan(8.0, 0.0, Ask(make="longer", add=10.0, marker=500.0))
    assert beyond.seconds == 8.0                                               # a marker past the end is the end
    with pytest.raises(ValueError, match="less than 1 second before the marker"):
        generate.plan(8.0, 0.0, Ask(make="longer", marker=0.5))


def test_variations_are_planned_as_before():
    plain = generate.plan(600.0, 200.0, Ask())
    assert (plain.start_s, plain.seconds, plain.whole, plain.part, plain.join_s, plain.out_s) == (200.0, 60.0, False, None, 0.0, 60.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_generate.py -q -k "passage_for_a_part or nothing_to_match or join_that_fits or follows_on or planned_as_before"`
Expected: FAIL, `generate` has no `plan`.

- [ ] **Step 3: Write `Plan` and `plan`**

After `passage` in `generate.py`:

```python
@dataclass(frozen=True)
class Plan:
    """The passage of a sample a run is made from, and what is made anew in it."""

    start_s: float
    seconds: float
    whole: bool  # the passage is the whole sample
    part: tuple[float, float] | None = None  # the new stretch, in seconds from the start of the passage
    join_s: float = 0.0  # a loop's join
    out_s: float = 0.0  # how long the clip will be


def plan(duration: float, matched_at: float, ask: Ask) -> Plan:
    """What a run does to a sample. A ValueError says, in words for the page, why it cannot."""
    if ask.make == "part":
        a, b = ask.span[0], min(ask.span[1], duration)
        if b - a < SHORTEST_PART_S:
            raise ValueError("The part lies past the end of the sample.")
        seconds = min(duration, LONGEST_S)
        if b - a > seconds - MIN_KEPT_S:
            raise ValueError(f"Mark a shorter part: at least {MIN_KEPT_S:g} second of the passage has to stay as it is.")
        start = min(max(0.0, (a + b) / 2 - seconds / 2), duration - seconds)
        return Plan(start, seconds, seconds >= duration - 0.05, (a - start, b - start), 0.0, seconds)
    if ask.make == "longer":
        marker = duration if ask.marker is None else min(ask.marker, duration)
        if marker < MIN_KEPT_S:
            raise ValueError(f"There is less than {MIN_KEPT_S:g} second before the marker to follow on from.")
        context = min(marker, LONGEST_S, LONGEST_ASKED_S - ask.add)
        return Plan(marker - context, context, context >= duration - 0.05, (context, context + ask.add), 0.0, context + ask.add)
    start, seconds = passage(duration, matched_at, ask.seconds)
    whole = seconds >= duration - 0.05
    if ask.make == "loop":
        if seconds < MIN_LOOP_S:
            raise ValueError(f"A loop needs at least {MIN_LOOP_S:g} seconds of sound.")
        return Plan(start, seconds, whole, None, min(JOINS[ask.join], seconds / 2), seconds)
    return Plan(start, seconds, whole, None, 0.0, seconds)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_generate.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/audio_embed/generate.py tests/test_generate.py
git commit -m "Choose the passage and the part for a redone part, a loop and a longer clip"
```

---

### Task 5: What the model is given, and its command line

**Files:**
- Modify: `src/audio_embed/generate.py` (imports at line 28, `command` at 126, new functions after `plan`)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `Plan` from Task 4; `splice.snap`, `splice.whole_steps`, `splice.turn`, `splice.turn_back`, `splice.join`, `splice.clock`, `splice.STEP`, `splice.SR` from Task 1.
- Produces:
  - `generate.lay_out(make: str, samples: np.ndarray, shape: Plan) -> tuple[np.ndarray, int, int, int, int]`: `(given, start, end, total, by)`. `given` is what Stable Audio 3 reads; `start` and `end` are the new stretch in it, in samples on steps; `total` is how many samples to ask for; `by` is how far a loop was turned (0 otherwise).
  - `generate.finish(make: str, given, made, start: int, end: int, by: int) -> np.ndarray`: the clip to keep.
  - `generate.command(sa3, ask, reference, seconds, seed, out, part=None)`: `part` is `(start, end, total)` in samples.

- [ ] **Step 1: Write the failing tests**

Add `from audio_embed import splice` to the imports at the top of `tests/test_generate.py`, then add:

```python
def level(n, value=100):
    return np.full((n, 2), value, dtype="<i2")


def test_a_part_is_laid_out_on_steps_inside_the_passage():
    samples = level(10 * 44100)
    given, start, end, total, by = generate.lay_out("part", samples, generate.Plan(0.0, 10.0, True, (3.0, 5.0), 0.0, 10.0))
    assert given is samples and by == 0 and total == len(samples)
    assert (start, end) == (splice.snap(3.0), splice.snap(5.0))
    assert start % splice.STEP == 0 and end % splice.STEP == 0


def test_a_part_that_reaches_the_end_of_a_one_shot_takes_its_whole_tail():
    samples = level(int(3.2 * 44100))   # not a whole number of steps
    _, start, end, total, _ = generate.lay_out("part", samples, generate.Plan(0.0, 3.2, True, (2.5, 3.2), 0.0, 3.2))
    assert end == total == len(samples)   # no sliver of the old tail is left after the part
    made = level(len(samples), 9000)
    out = generate.finish("part", samples, made, start, end, 0)
    assert len(out) == len(samples) and np.array_equal(out[start:], made[start:])


def test_a_loop_is_turned_so_its_ends_meet_inside_the_join():
    samples = np.repeat(np.arange(60 * 4096 + 500, dtype="<i4")[:, None] % 30000, 2, axis=1).astype("<i2")
    given, start, end, total, by = generate.lay_out("loop", samples, generate.Plan(0.0, 5.58, True, None, 2.0, 5.58))
    assert total == len(given) == 60 * 4096      # trimmed to whole steps
    assert by == 30 * 4096
    seam = total - by                              # where the old end meets the old start
    assert start < seam < end and (end - start) == 22 * 4096   # two seconds is 21.5 steps
    assert given[seam - 1].tolist() == samples[total - 1].tolist() and given[seam].tolist() == samples[0].tolist()

    made = level(total, 9000)
    out = generate.finish("loop", given, made, start, end, by)
    assert len(out) == total
    # Turned back: the join is at the two ends, and the middle is the sample as it was.
    assert out[0, 0] == 9000 and out[-1, 0] == 9000
    middle = slice(by - 2 * 4096, by + 2 * 4096)
    assert np.array_equal(out[middle], samples[:total][middle])


def test_a_join_never_takes_more_than_half_a_short_loop():
    samples = level(22 * 4096)   # about two seconds
    _, start, end, total, _ = generate.lay_out("loop", samples, generate.Plan(0.0, 2.04, True, None, 4.0, 2.04))
    assert end - start == 11 * 4096 and 0 < start and end < total


def test_longer_asks_for_the_passage_and_what_follows_it():
    samples = level(8 * 44100)
    given, start, end, total, by = generate.lay_out("longer", samples, generate.Plan(0.0, 8.0, True, (8.0, 18.0), 0.0, 18.0))
    assert len(given) == len(samples) // 4096 * 4096 and by == 0
    assert start == len(given) and end == total == len(given) + splice.snap(10.0)
    out = generate.finish("longer", given, level(total, 9000), start, end, by)
    assert len(out) == total and out[-1, 0] == 9000 and out[0, 0] == 100


def test_command_for_a_part_names_the_stretch_and_not_a_distance():
    sa3 = Path("/sa3")
    cmd = generate.command(sa3, Ask(prompt="bells", make="part"), Path("/tmp/ref.wav"), 10.0, 7, Path("/tmp/out.wav"),
                           part=(12288, 20480, 45056))
    assert cmd[cmd.index("--init-audio") + 1] == "/tmp/ref.wav"
    assert cmd[cmd.index("--inpaint-range") + 1] == "0.278638,0.464398"
    assert cmd[cmd.index("--seconds") + 1] == "1.021677"
    assert "--init-noise-level" not in cmd
    assert cmd[-4:] == ["--seed", "7", "--out", "/tmp/out.wav"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_generate.py -q -k "laid_out or whole_tail or ends_meet or half_a_short or what_follows or names_the_stretch"`
Expected: FAIL, `generate` has no `lay_out`.

- [ ] **Step 3: Write `lay_out`, `finish` and the new `command`**

Change the import line to `from . import audio, renders, splice` and add `import numpy as np` with the other imports.

After `plan`:

```python
def lay_out(make: str, samples: np.ndarray, shape: Plan) -> tuple[np.ndarray, int, int, int, int]:
    """(what Stable Audio 3 is given, the start and end of the new stretch in it, the samples to ask for, how far a loop was turned)."""
    if make == "part":
        n = len(samples)
        start, end = splice.snap(shape.part[0]), min(splice.snap(shape.part[1]), n)
        if n - end < splice.STEP:
            end = n  # reaching the end means all of it: no sliver of the old tail after the part
        return samples, max(0, min(start, end - splice.STEP)), end, n, 0
    whole = splice.whole_steps(samples)
    if make == "loop":
        steps = len(whole) // splice.STEP
        by = steps // 2 * splice.STEP
        join = min(max(2, round(shape.join_s * splice.SR / splice.STEP)), steps // 2)
        start = len(whole) - by - join // 2 * splice.STEP
        return splice.turn(whole, by), start, start + join * splice.STEP, len(whole), by
    add = max(splice.STEP, splice.snap(shape.part[1] - shape.part[0]))
    return whole, len(whole), len(whole) + add, len(whole) + add, 0


def finish(make: str, given: np.ndarray, made: np.ndarray, start: int, end: int, by: int) -> np.ndarray:
    """The clip to keep: the new stretch joined into what was given, and a loop turned back."""
    joined = splice.join(given, made, start, end)
    return splice.turn_back(joined, by) if make == "loop" else joined
```

Replace `command` with:

```python
def command(sa3: Path, ask: Ask, reference: Path | None, seconds: float, seed: int, out: Path,
            part: tuple[int, int, int] | None = None) -> list[str]:
    """The Stable Audio 3 command line for one clip.

    Without a `reference` the clip is made from the prompt alone. With a `part`
    (start, end, samples in all) only that stretch of the reference is made anew.
    """
    dit, decoder = MODELS[ask.model]
    cmd = [
        str(sa3 / ".venv/bin/python"), str(sa3 / "scripts/sa3_mlx.py"),
        "--prompt", ask.prompt, "--dit", dit, "--decoder", decoder,
    ]
    if reference is not None and part is not None:
        start, end, total = part
        cmd += ["--init-audio", str(reference), "--inpaint-range", f"{splice.clock(start)},{splice.clock(end)}",
                "--seconds", splice.clock(total)]
    else:
        if reference is not None:
            cmd += ["--init-audio", str(reference), "--init-noise-level", str(DISTANCES[ask.distance])]
        cmd += ["--seconds", str(round(seconds, 2))]
    cmd += ["--seed", str(seed), "--out", str(out)]
    if ask.strength != 1.0:
        cmd += ["--cfg", str(ask.strength)]
    if ask.avoid:
        cmd += ["--negative-prompt", ask.avoid]
    return cmd
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_generate.py -q`
Expected: all pass, including `test_command_runs_the_checkout_with_its_own_python` and `test_command_from_text_alone_names_no_sample`, which pin the old command lines.

- [ ] **Step 5: Commit**

```bash
git add src/audio_embed/generate.py tests/test_generate.py
git commit -m "Lay out a part, a loop and a longer clip for Stable Audio 3"
```

---

### Task 6: A run that makes a part, a loop or a longer clip

**Files:**
- Modify: `src/audio_embed/generate.py` (`Generator.start` at 238, `start_from_text` at 248, `_queue_run` at 254, `_make` at 288)
- Modify: `tests/test_generate.py` (`stand_in` at line 34)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `plan`, `Plan`, `lay_out`, `finish`, `command(..., part=)` from Tasks 4 and 5.
- Produces:
  - `Generator.start(source, key, name, duration, start_s, ask)` unchanged in signature; raises `ValueError` from `plan`.
  - `Generator._queue_run(source, key, name, shape: Plan, ask, parent: str | None = None) -> str`
  - A clip's story gains `"make"`, `"part"` (`[start, end]` in seconds in the clip, or `None`), `"join"` (seconds, or `None`), `"add"` (seconds, or `None`), `"parent"` (`str | None`) and `"length"` (the clip's own length in seconds). `"seconds"` stays the length of the passage.
  - A clip in `progress()` has `"seconds"` set to its own length once it is done.

- [ ] **Step 1: Teach the stand-in to regenerate a stretch**

In `tests/test_generate.py`, replace the body of `run` in `stand_in` from `out = command[...]` down to the `return` with:

```python
        out = command[command.index("--out") + 1]
        seconds = float(command[command.index("--seconds") + 1])
        if "--inpaint-range" in command:  # a level no sample has, as long as was asked for (or `short` samples less)
            splice.write(out, np.full((int(round(seconds * 44100)) - short, 2), 9000, dtype="<i2"))
        elif "--init-audio" in command:
            shutil.copy(command[command.index("--init-audio") + 1], out)
        else:  # from text alone: a clip of the asked length
            write_wav(out, seconds=seconds, sr=44100, channels=2)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
```

and change its signature to `def stand_in(calls=None, fails=False, short=0):`.

- [ ] **Step 2: Write the failing tests**

```python
def story_of(gen, clip):
    return gen.story(clip["id"])


def test_a_redone_part_is_the_sample_with_only_that_part_new(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=4.0, sr=44100, channels=2)
    (clip,) = made(gen, source, Ask(count=1, make="part", span=(1.0, 2.0)), duration=4.0)
    assert clip["state"] == "done"

    command = calls[0][0]
    start, end = splice.snap(1.0), splice.snap(2.0)
    assert command[command.index("--inpaint-range") + 1] == f"{splice.clock(start)},{splice.clock(end)}"
    out, original = splice.read(gen.clip(clip["id"])), splice.read(source)
    assert len(out) == len(original)
    assert np.all(out[start:end] == 9000)
    assert np.array_equal(out[: start - splice.FADE], original[: start - splice.FADE])
    assert np.array_equal(out[end + splice.FADE:], original[end + splice.FADE:])

    story = story_of(gen, clip)
    assert (story["make"], story["join"], story["add"], story["parent"]) == ("part", None, None, None)
    assert story["part"] == [round(start / 44100, 2), round(end / 44100, 2)]
    assert story["length"] == pytest.approx(4.0, abs=0.01) and story["seconds"] == 4.0


def test_a_loop_is_new_at_both_ends_and_untouched_in_the_middle(tmp_path):
    gen = generator(tmp_path)
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=4.0, sr=44100, channels=2)
    (clip,) = made(gen, source, Ask(count=1, make="loop", join="medium"), duration=4.0)
    out, original = splice.read(gen.clip(clip["id"])), splice.read(source)
    assert len(out) == len(original) // 4096 * 4096
    assert out[0, 0] == 9000 and out[-1, 0] == 9000
    middle = slice(len(out) // 2 - 4096, len(out) // 2 + 4096)
    assert np.array_equal(out[middle], original[middle])
    story = story_of(gen, clip)
    assert (story["make"], story["part"]) == ("loop", None)
    assert story["join"] == pytest.approx(2.0, abs=0.1)
    assert clip["seconds"] == pytest.approx(len(out) / 44100)


def test_a_longer_clip_is_the_sample_and_then_new_sound(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=3.0, sr=44100, channels=2)
    (clip,) = made(gen, source, Ask(count=1, make="longer", add=2.0), duration=3.0)
    out, original = splice.read(gen.clip(clip["id"])), splice.read(source)
    kept = len(original) // 4096 * 4096
    assert len(out) == kept + splice.snap(2.0)
    assert np.array_equal(out[: kept - splice.FADE], original[: kept - splice.FADE])
    assert np.all(out[kept:] == 9000)
    command = calls[0][0]
    assert command[command.index("--seconds") + 1] == splice.clock(len(out))
    story = story_of(gen, clip)
    assert story["make"] == "longer" and story["add"] == pytest.approx(2.0, abs=0.1)
    assert clip["seconds"] == pytest.approx(len(out) / 44100)


def test_a_clip_that_comes_back_too_short_fails_and_the_run_goes_on(tmp_path):
    gen = generator(tmp_path, run=stand_in(short=30000))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=3.0, sr=44100, channels=2)
    clips = made(gen, source, Ask(count=2, make="longer", add=1.0), duration=3.0)
    assert [clip["state"] for clip in clips] == ["failed", "failed"]
    assert clips[0]["error"] == "Stable Audio 3 returned a shorter clip than was asked for."
    assert gen.clips_from(str(source)) == []


def test_a_run_that_cannot_be_planned_is_refused_before_it_starts(tmp_path):
    gen = generator(tmp_path)
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=1.0)
    with pytest.raises(ValueError, match="at least 2 seconds"):
        gen.start(source, key=str(source), name="tone", duration=1.0, start_s=0.0, ask=Ask(make="loop"))
    with pytest.raises(ValueError, match="from text alone"):
        gen.start_from_text(Ask(prompt="rain", make="loop"))
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/test_generate.py -q -k "only_that_part_new or new_at_both_ends or then_new_sound or too_short_fails or cannot_be_planned"`
Expected: FAIL (the clips come back as plain copies, and the stories have no `make`).

- [ ] **Step 4: Change the run**

In `generate.py`, replace `start`, `start_from_text`, `_queue_run` and `_make` with:

```python
    def start(self, source: Path, key: str, name: str, duration: float, start_s: float, ask: Ask) -> str:
        """Queue a run of `ask.count` clips and give its id. A ValueError says why the run cannot be made.

        `source` is the file to read, `key` the library path the clips belong to
        (they differ when an online-only file is read from its render) and `name`
        the sample's name without its suffix.
        """
        return self._queue_run(Path(source), key, name, plan(duration, start_s, ask), ask)

    def start_from_text(self, ask: Ask) -> str:
        """Queue a run of clips made from the prompt alone, with no sample, and give its id."""
        if ask.make != "variations":
            raise ValueError("Only whole new clips can be made from text alone. Work from a clip to change it.")
        if not ask.prompt:
            raise ValueError("Type what to generate first.")
        seconds = ask.seconds or TEXT_SECONDS
        return self._queue_run(None, FROM_TEXT, "", Plan(0.0, seconds, True, None, 0.0, seconds), ask)

    def _queue_run(self, source: Path | None, key: str, name: str, shape: Plan, ask: Ask, parent: str | None = None) -> str:
        run = {
            "id": secrets.token_hex(8), "source": source, "key": key, "name": name,
            "plan": shape, "ask": ask, "parent": parent, "done": False,
            "clips": [
                {"id": secrets.token_hex(8), "n": n, "state": "waiting", "seconds": shape.out_s, "error": None}
                for n in range(1, ask.count + 1)
            ],
        }
        with self._lock:
            self._runs[run["id"]] = run
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, daemon=True)
                self._worker.start()
        self._queue.put(run)
        return run["id"]
```

```python
    def _make(self, run: dict) -> None:
        ask: Ask = run["ask"]
        shape: Plan = run["plan"]
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            reference, given, part, by = None, None, None, 0
            if run["source"] is not None:
                reference = Path(scratch) / "reference.wav"
                cut(run["source"], shape.start_s, shape.seconds, reference)
                if ask.make != "variations":
                    # Stable Audio 3 is given the passage as it will be joined into: trimmed to steps, a loop turned.
                    given, start, end, total, by = lay_out(ask.make, splice.read(reference), shape)
                    splice.write(reference, given)
                    part = (start, end, total)
            for clip in run["clips"]:
                clip["state"] = "making"
                seed = secrets.randbelow(2**31)
                out = Path(scratch) / f"{clip['id']}.wav"
                result = self._run(
                    command(self.sa3, ask, reference, shape.out_s, seed, out, part),
                    cwd=self.sa3, capture_output=True, text=True,
                )
                if result.returncode or not out.is_file():
                    said = [line for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
                    clip.update(state="failed", error=(said[-1].strip() if said else "Stable Audio 3 made no file.")[:300])
                    continue
                length = shape.out_s
                if part is not None:
                    try:
                        kept = finish(ask.make, given, splice.read(out), part[0], part[1], by)
                    except ValueError as e:
                        clip.update(state="failed", error=str(e)[:300])
                        continue
                    splice.write(out, kept)
                    length = len(kept) / splice.SR
                new = part is not None and ask.make != "loop"
                story = {
                    "clip": clip["id"], "n": clip["n"], "key": run["key"], "name": run["name"],
                    "start_s": shape.start_s, "seconds": shape.seconds, "whole": shape.whole, "length": length,
                    "make": ask.make, "parent": run["parent"],
                    "part": [round(part[0] / splice.SR, 2), round(min(part[1], len(kept)) / splice.SR, 2)] if new else None,
                    "join": round((part[1] - part[0]) / splice.SR, 2) if ask.make == "loop" else None,
                    "add": round((part[1] - part[0]) / splice.SR, 2) if ask.make == "longer" else None,
                    "prompt": ask.prompt, "avoid": ask.avoid, "distance": ask.distance, "strength": ask.strength,
                    "model": ask.model, "seed": seed, "made": time.time(), "kept": None,
                }
                # The story first, so a clip in the cache always has one.
                self._story_file(clip["id"]).write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")
                shutil.move(out, self.cache / f"{clip['id']}.wav")
                clip.update(state="done", seconds=length)
                self._trim({c["id"] for c in run["clips"]})
```

Note for the implementer: `kept` is only read inside `if new`, which is only true when `part is not None`, where `kept` has been set. `"seconds"` in the story is now the passage's length (`shape.seconds`); for variations and text that is the clip's length, as before.

- [ ] **Step 5: Run all the generator tests**

Run: `uv run pytest tests/test_generate.py -q`
Expected: all pass. `test_a_run_makes_its_clips_one_per_seed_from_the_asked_passage` still passes: its clips keep `"seconds": 2.0`.

- [ ] **Step 6: Commit**

```bash
git add src/audio_embed/generate.py tests/test_generate.py
git commit -m "Make a part, a loop or a longer clip in a run, keeping the rest of the sample exact"
```

---

### Task 7: Describing a chain, and working from a clip

**Files:**
- Modify: `src/audio_embed/generate.py` (`describe` at 154; new `Generator.start_from_clip` after `start_from_text`; add `import wave`)
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: the story fields from Task 6; `Generator.clip`, `Generator.story`, `_queue_run(..., parent=)`.
- Produces:
  - `describe(made: dict) -> str` reads `make`, `part`, `join`, `add`, `parent`; a story without `make` reads as a variation.
  - `Generator.start_from_clip(clip_id: str, ask: Ask) -> str`: queues a run whose sample is a clip in the cache; its clips get the same `key` and `name` as that clip and `parent` set to its description. Raises `ValueError` if the clip is gone or the run cannot be planned.

- [ ] **Step 1: Write the failing tests**

```python
BASE = {"name": "Pad 04", "start_s": 0.0, "seconds": 20.0, "whole": True, "prompt": "", "avoid": "",
        "distance": "medium", "strength": 1.0, "model": "medium", "seed": 7}


def test_description_says_what_was_made_anew():
    assert generate.describe({**BASE, "make": "part", "part": [12.0, 15.0], "prompt": "bells"}) == (
        'Generated from "Pad 04". 0:12 to 0:15 regenerated. Prompt: bells. Model: Stable Audio 3 medium. Seed: 7.')
    assert generate.describe({**BASE, "make": "loop", "join": 2.0}) == (
        'Generated from "Pad 04". Made to loop (2 s join). Model: Stable Audio 3 medium. Seed: 7.')
    assert generate.describe({**BASE, "make": "longer", "part": [20.0, 50.0], "add": 30.0}) == (
        'Generated from "Pad 04". Continued for 30 s from 0:20. Model: Stable Audio 3 medium. Seed: 7.')


def test_description_tells_the_whole_chain():
    first = generate.describe({**BASE, "distance": "far", "seed": 3})
    then = generate.describe({**BASE, "make": "loop", "join": 2.0, "seed": 9, "parent": first})
    assert then == ('Generated from "Pad 04". Distance: far. Model: Stable Audio 3 medium. Seed: 3.'
                    " Then made to loop (2 s join). Model: Stable Audio 3 medium. Seed: 9.")
    again = generate.describe({**BASE, "distance": "close", "seed": 4, "parent": then})
    assert again.endswith("Then a variation of it, distance close. Model: Stable Audio 3 medium. Seed: 4.")


def test_a_story_from_before_this_change_is_a_variation():
    assert generate.describe(dict(BASE)) == 'Generated from "Pad 04". Distance: medium. Model: Stable Audio 3 medium. Seed: 7.'


def test_a_run_can_start_from_a_clip_and_is_filed_with_it(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=4.0, sr=44100, channels=2)
    (first,) = made(gen, source, Ask(count=1, prompt="shimmer"), duration=4.0)

    run = gen.start_from_clip(first["id"], Ask(count=1, make="loop"))
    for _ in range(500):
        if gen.progress(run)["done"]:
            break
        time.sleep(0.01)
    (second,) = gen.progress(run)["clips"]
    assert second["state"] == "done"
    # It was cut from the clip in the cache, not from the library file.
    assert calls[-1][0][calls[-1][0].index("--init-audio") + 1] != str(source)
    story = gen.story(second["id"])
    assert (story["key"], story["name"], story["make"]) == (str(source), "tone", "loop")
    assert story["parent"] == generate.describe(gen.story(first["id"]))
    assert [c["id"] for c in gen.clips_from(str(source))] == [first["id"], second["id"]]
    assert "Then made to loop" in generate.describe(story)


def test_a_clip_that_is_gone_cannot_be_worked_from(tmp_path):
    gen = generator(tmp_path)
    with pytest.raises(ValueError, match="no longer in the cache"):
        gen.start_from_clip("0123456789abcdef", Ask())
    with pytest.raises(ValueError, match="no longer in the cache"):
        gen.start_from_clip("../../etc/passwd", Ask())
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_generate.py -q -k "made_anew or whole_chain or before_this_change or start_from_a_clip or is_gone"`
Expected: FAIL.

- [ ] **Step 3: Rewrite `describe` and add `start_from_clip`**

Replace `describe` with:

```python
def describe(made: dict) -> str:
    """Where a clip came from, in a sentence or two, for its comment tag. A clip made from a clip tells the whole chain."""

    def clock(s):
        return f"{int(s // 60)}:{int(s % 60):02d}"

    make, parent, part = made.get("make", "variations"), made.get("parent"), made.get("part")
    from_text = made.get("key") == FROM_TEXT
    if make == "part":
        step = f"{clock(part[0])} to {clock(part[1])} regenerated."
    elif make == "loop":
        step = f"made to loop ({made['join']:g} s join)."
    elif make == "longer":
        step = f"continued for {made['add']:g} s from {clock(part[0])}."
    else:
        step = f"a variation of it, distance {made['distance']}." if parent else ""
    if parent:
        parts = [parent, f"Then {step}"]
    else:
        whole = made.get("whole", made["start_s"] == 0)
        at = "" if whole else f" ({clock(made['start_s'])} to {clock(made['start_s'] + made['seconds'])})"
        parts = ["Generated from text." if from_text else f'Generated from "{made["name"]}"{at}.']
        if step:
            parts.append(step[0].upper() + step[1:])
    if made["prompt"]:
        parts.append(f"Prompt: {made['prompt'].rstrip('.')}.")
    if made["avoid"]:
        parts.append(f"Avoid: {made['avoid'].rstrip('.')}.")
    if make == "variations" and not from_text and not parent:
        parts.append(f"Distance: {made['distance']}.")
    if made["strength"] != 1:
        parts.append(f"Prompt strength: {made['strength']:g}.")
    parts.append(f"Model: Stable Audio 3 {made['model']}. Seed: {made['seed']}.")
    return " ".join(parts)
```

Add `import wave` to the imports, and after `start_from_text`:

```python
    def start_from_clip(self, clip_id: str, ask: Ask) -> str:
        """Queue a run whose sample is a clip in the cache. Its clips are filed with the clip they came from."""
        clip, story = self.clip(clip_id), self.story(clip_id)
        if clip is None or story is None:
            raise ValueError("That clip is no longer in the cache. Generate it again.")
        with wave.open(str(clip), "rb") as f:
            duration = f.getnframes() / f.getframerate()
        return self._queue_run(clip, story["key"], story["name"], plan(duration, 0.0, ask), ask, parent=describe(story))
```

`Generator.clip` already refuses anything that is not 16 hex digits, so `clip_id` is safe to take from the page.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_generate.py -q`
Expected: all pass, including `test_description_says_where_a_clip_came_from` and `test_a_kept_clip_made_from_text_is_named_and_described_by_its_prompt` from before.

- [ ] **Step 5: Commit**

```bash
git add src/audio_embed/generate.py tests/test_generate.py
git commit -m "Work from a clip that has not been kept, and describe the chain it came by"
```

---

### Task 8: The routes

The routes are closures inside `create_app` and have no tests of their own; everything they decide is in `generate.py`, which Tasks 3 to 7 cover. Keep them thin. They are checked by hand in Task 12.

**Files:**
- Modify: `src/audio_embed/server.py` (`clip_json` at 606, `generated_from` at 614, `generate_start` at 633)

**Interfaces:**
- Consumes: `Generator.start` (raises `ValueError`), `Generator.start_from_clip`, `Generator.story`, story fields `make`, `parent`, `length`.
- Produces:
  - A clip's JSON gains `"make"`; its `"seconds"` is the clip's own length; `"distance"` is `null` unless the clip is a variation of something.
  - `GET /api/generated?from=ID&clip=CLIP` (or `?text=1&clip=CLIP`) adds `"working_from"` (that clip's JSON, or `null`) and `"gone"` (`true` when a clip was asked for and is not there).
  - `POST /api/generate` body may carry `"clip"`, `"make"`, `"span"`, `"marker"`, `"join"`, `"add"`.

- [ ] **Step 1: Replace `clip_json`**

```python
    def clip_json(story: dict) -> dict:
        make = story.get("make", "variations")
        # Only a variation of a sample or of a clip is near or far from something.
        varied = make == "variations" and (story["key"] != generate.FROM_TEXT or story.get("parent"))
        return {
            "id": story["clip"], "n": story["n"], "seconds": story.get("length", story["seconds"]),
            "prompt": story["prompt"], "make": make, "distance": story["distance"] if varied else None,
            "kept": Path(story["kept"]).name if story.get("kept") else None,
        }
```

- [ ] **Step 2: Let the view be told which clip it works from**

In `generated_from`, after the `ready = generator.models()` line add:

```python
        # A clip of this view that the panel should work from in place of the file.
        working_from, asked = None, request.query_params.get("clip")
        if asked:
            story = generator.story(asked)
            working_from = clip_json(story) if story is not None and story["key"] == key else None
```

and add to the returned JSON:

```python
            "working_from": working_from,
            "gone": bool(asked) and working_from is None,
```

- [ ] **Step 3: Start runs from a clip, and say why a run cannot be planned**

Replace `generate_start` with:

```python
    async def generate_start(request):
        """Start making clips: {"id", "start", "prompt", "avoid", "distance", "count", "make", ...}.

        With "clip" they are made from that clip in the cache; with neither "id" nor "clip", from the prompt alone.
        """
        body = await request.json()
        clip_id = body.get("clip")
        from_text = body.get("id") is None and not clip_id
        try:
            found = None if from_text or clip_id else library.entry(int(body["id"]))
            start = float(body.get("start") or 0)
            ask = generate.ask_from(body)
        except (TypeError, ValueError) as e:
            return error(str(e), 400)
        if found is None and not from_text and not clip_id:
            return error("That file is not in the index.", 400)
        why = generator.problem(ask.model)
        if why:
            return error(why, 409)
        try:
            if clip_id:
                return JSONResponse({"run": generator.start_from_clip(str(clip_id), ask)})
            if from_text:
                return JSONResponse({"run": generator.start_from_text(ask)})
            _, matrix, position = found
            path = Path(matrix.paths[position])
            _, source = library.heard_from(path)
            if source is None:
                return error("The original is online-only and its render is not within reach, so there is nothing to generate from.", 409)
            return JSONResponse({"run": generator.start(source, str(path), path.stem, matrix.durations[position], start, ask)})
        except ValueError as e:
            return error(str(e), 400)
```

- [ ] **Step 4: Check nothing else broke**

Run: `uv run pytest -q`
Expected: all pass.

Run: `uv run python -c "from audio_embed import server; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/audio_embed/server.py
git commit -m "Pass what to make, and the clip to work from, through the generate routes"
```

---

### Task 9: The Make row on the page

**Files:**
- Modify: `src/audio_embed/web/index.html` (constants near line 736, `gen` at 316, `clipHit` at 768, `showGenerate` at 795, `load` at 1152, CSS near line 186)

**Interfaces:**
- Consumes: the routes of Task 8.
- Produces (used by Tasks 10 and 11): `gen.make`, `gen.join`, `gen.add`, `gen.span`, `gen.marker`; inside `showGenerate`, the names `working`, `base`, `noSample`, `showSample`, and a function `markOn(item)` that Task 10 fills in.

There are no automated tests for the page. Each step says what to look at; the full hand test is Task 12.

- [ ] **Step 1: Add the choices and the state**

After the `STRENGTHS` line add:

```js
const MAKES = [['variations', 'Variations'], ['part', 'Redo a part'], ['loop', 'Loop'], ['longer', 'Longer']];
const JOINS = [['short', 'Short'], ['medium', 'Medium'], ['long', 'Long']];
const ADDS = [[10, '10 s'], [30, '30 s'], [60, '1 min']];
// What a clip's row is called by what was made.
const MADE = { part: 'Redone', loop: 'Loop', longer: 'Longer' };
```

Change the `gen` line to:

```js
const gen = { prompt: '', avoid: '', distance: 'medium', more: false, count: 4, seconds: '', textSeconds: 10, model: 'medium', strength: 1,
  make: 'variations', join: 'medium', add: 30, span: null, marker: null, of: null };
```

Add to the CSS after the `.gen-more[hidden]` rule:

```css
  .gen .filter[hidden] { display: none; }
  .gen-hint { color: var(--dim); font-size: 13px; }
  .gen-hint:empty { display: none; }
```

- [ ] **Step 2: Name clips by what was made, and repeat a loop**

Replace `clipHit` with:

```js
const clipHit = (clip, number) => ({
  id: `clip-${clip.id}`, clip: clip.id, kept: clip.kept, folder: '',
  name: `${MADE[clip.make] || (clip.distance ? 'Variation' : 'Clip')} ${number}`,
  near: [MADE[clip.make]?.toLowerCase() || DISTANCES.find(([value]) => value === clip.distance)?.[1].toLowerCase(), clip.prompt].filter(Boolean).join(' · '),
  duration: clip.seconds, window: clip.seconds, start: 0, figure: '', audio: 'original', loops: clip.make === 'loop',
  src: `/generated/${clip.id}`, peaksUrl: `/api/generated/${clip.id}/peaks`,
});
```

In `load`, add as its first line:

```js
  player.loop = !!hit.loops;   // a loop plays on repeat, so its join is heard
```

- [ ] **Step 3: Show the Make row and the rows that go with each choice**

In `showGenerate`, replace everything from the line `// The sample itself, with the passage the clips will be made from marked on it.` down to and including the line `if (far) far.title = 'How far the clips may drift from the sample.';` with:

```js
    // The clip the panel works from in place of the file, if one was asked for (Task 11 sets query.clip).
    const working = body.working_from;
    const base = working ? clipHit(working, working.n) : body.source;
    const noSample = fromText && !working;
    const of = working ? working.id : query.id;
    if (gen.of !== of) { gen.of = of; gen.span = null; gen.marker = null; }   // a part belongs to one sample
    if (noSample) gen.make = 'variations';

    // The sample itself, with the passage the clips will be made from marked on it.
    const sample = el('ol', { className: 'hits' });
    const showSample = () => {
      if (noSample) return;
      const whole = gen.make === 'part' || gen.make === 'longer';   // these choose their own passage
      const [from, seconds] = whole ? [0, base.duration] : genPassage(base.duration, working ? 0 : query.start || 0);
      const item = row(listed({ ...base }, {
        start: from, window: seconds, at: 'uses from',
        part: gen.make === 'part' ? gen.span : null,
        marker: gen.make === 'longer' ? gen.marker ?? base.duration : null,
      }), 1, null);
      sample.replaceChildren(item);
      waveforms([item]);
      markOn(item);
    };

    const clips = el('ol', { className: 'hits' });
    const says = el('span', { className: 'gen-says', role: 'status' });
    const go = el('button', { type: 'submit' });
    const label = () => { go.textContent = `Generate ${gen.count}`; };
    label();
    const model = body.models.length > 1 ? [genChoice('Model', body.models.map((m) => [m, GEN_MODELS[m] || m]), 'model')] : [];
    const far = genChoice('How far', DISTANCES, 'distance');
    far.title = 'How far the clips may drift from the sample.';
    const join = genChoice('Join', JOINS, 'join');
    join.title = 'How much of the end and the start is made anew so that they meet: 1, 2 or 4 seconds.';
    const adds = genChoice('Add', ADDS, 'add');
    const lengths = genChoice('Length', CLIP_LENGTHS, 'seconds', showSample);
    const hint = el('span', { className: 'gen-hint' });
    // Each choice of what to make has its own rows; the rest are put away.
    const fit = () => {
      far.hidden = gen.make !== 'variations';
      join.hidden = gen.make !== 'loop';
      adds.hidden = gen.make !== 'longer';
      lengths.hidden = gen.make === 'part' || gen.make === 'longer';
      hint.textContent = {
        part: 'Drag on the waveform above to mark the part to redo.',
        longer: 'New sound follows on from the end. Click the waveform above to start it earlier, before a fade.',
      }[gen.make] || '';
      showSample();
    };
    const makes = genChoice('Make', MAKES, 'make', () => { fit(); keepAddress(); });
    // From text, the length and the model decide the most, so they are not folded away.
    const more = el('div', { className: 'gen-more', hidden: !gen.more },
      genText('avoid', 'Avoid', 'What to steer away from, such as: drums, vocals'),
      genChoice('Count', COUNTS, 'count', label),
      ...(noSample ? [] : [lengths, ...model]),
      genChoice('Strength', STRENGTHS, 'strength'));
    more.querySelector('[data-row="Strength"]').title = 'How hard the prompt pulls. Anything to avoid needs more than Normal, so it is raised to Strong if left there.';
    const fold = el('button', { className: 'plain', type: 'button', ariaExpanded: String(gen.more) }, gen.more ? 'Less' : 'More');
    fold.onclick = () => { gen.more = !gen.more; more.hidden = !gen.more; fold.ariaExpanded = String(gen.more); fold.textContent = gen.more ? 'Less' : 'More'; };
    const panel = el('form', { className: 'gen' },
      ...(noSample
        ? [genText('prompt', 'Prompt', 'The sound to make, such as: a dark, ominous low drone'), genChoice('Length', TEXT_LENGTHS, 'textSeconds'), ...model]
        : [makes, genText('prompt', 'Prompt', 'What to add or change, such as: slow metallic shimmer'), far, join, adds, hint]),
      more,
      el('div', { className: 'gen-go' }, go, fold, says));
    fit();
```

Add this function just above `showGenerate` (Task 10 replaces its body):

```js
// Marking a part or a marker on the sample's waveform.
function markOn(item) {}
```

- [ ] **Step 4: Send the choices with the request**

In the same function, change the line in `follow` that adds finished clips to:

```js
      progress.clips.filter((clip) => clip.state === 'done').forEach((clip) => add({ ...clip, prompt: gen.sent.prompt, distance: gen.sent.distance, make: gen.sent.make }));
```

Replace the `gen.sent = ...` line and the `fetch('/api/generate', ...)` call's `body:` with:

```js
      const varied = !noSample && gen.make === 'variations';
      gen.sent = { prompt: gen.prompt.trim().replace(/\s+/g, ' '), distance: varied ? gen.distance : null, make: gen.make };
      fetch('/api/generate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(noSample
          ? { prompt: gen.prompt, avoid: gen.avoid, count: gen.count, seconds: gen.textSeconds, model: gen.model, strength: gen.strength }
          : { ...(working ? { clip: working.id } : { id: query.id, start: query.start || 0 }),
            make: gen.make, span: gen.make === 'part' ? gen.span : null, marker: gen.make === 'longer' ? gen.marker : null,
            join: gen.join, add: gen.add, prompt: gen.prompt, avoid: gen.avoid, distance: gen.distance,
            count: gen.count, seconds: gen.seconds || null, model: gen.model, strength: gen.strength }),
```

In the last lines of the function, replace `if (fromText) column.append(panel, clips);` with `if (noSample) column.append(panel, clips);`, and `else if (body.source.audio === 'none') {` with `else if (!working && body.source.audio === 'none') {`.

- [ ] **Step 5: Look at it (no model needed)**

Start a server on a scratch index as "Testing the page" in `CLAUDE.md` describes, open a file's generate view, and check without pressing Generate:
- The Make row shows four choices; Variations shows How far; Loop shows Join; Longer shows Add and its hint; Redo a part shows its hint.
- Length under More is hidden for Redo a part and Longer.
- Generate from text shows no Make row.
- Pressing Generate with Redo a part and nothing marked shows "Mark the part to redo on the waveform first."

- [ ] **Step 6: Commit**

```bash
git add src/audio_embed/web/index.html
git commit -m "Add a Make row to the generate panel: variations, a part, a loop, longer"
```

---

### Task 10: Marking the part and the marker on the waveform

**Files:**
- Modify: `src/audio_embed/web/index.html` (`markOn` from Task 9; `draw` at 1108)

**Interfaces:**
- Consumes: `gen.make`, `gen.span`, `gen.marker`; a row's `item._hit` with `part` and `marker` set by `showSample` (Task 9); `draw(item)`.
- Produces: `gen.span = [start, end]` in seconds in the sample after a drag; `gen.marker` in seconds after a click in Longer.

- [ ] **Step 1: Draw the part and the marker**

In `draw`, just before the line `if (current && current.id === hit.id) {`, add:

```js
  if (hit.part) {
    const a = (hit.part[0] / hit.duration) * w, b = (hit.part[1] / hit.duration) * w;
    g.globalAlpha = 0.3; g.fillStyle = hue; g.fillRect(a, 0, Math.max(b - a, 2), h);
    g.globalAlpha = 1; g.fillRect(a, 0, 1.5, h); g.fillRect(Math.max(a, b - 1.5), 0, 1.5, h);
  }
  if (hit.marker != null) {
    const x = Math.min(w - 2, (hit.marker / hit.duration) * w);
    g.globalAlpha = 0.3; g.fillStyle = hue; g.fillRect(x, 0, w - x, h);   // what new sound replaces
    g.globalAlpha = 1; g.fillRect(x, 0, 2, h);
  }
```

- [ ] **Step 2: Fill in `markOn`**

Replace the empty `markOn` with:

```js
// Marking on the sample's waveform: a drag marks the part to redo, and for Longer a click
// sets where new sound takes over. A plain click on a part still plays from there.
function markOn(item) {
  if (gen.make !== 'part' && gen.make !== 'longer') return;
  const canvas = item.querySelector('canvas'), hit = item._hit;
  const play = canvas.onclick;
  const at = (e) => {
    const box = canvas.getBoundingClientRect();
    return Math.round(Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)) * hit.duration * 100) / 100;
  };
  let from = null, dragged = false;
  canvas.style.cursor = gen.make === 'part' ? 'crosshair' : 'col-resize';
  canvas.onpointerdown = (e) => {
    if (gen.make !== 'part') return;
    from = at(e);
    dragged = false;
    canvas.setPointerCapture(e.pointerId);
  };
  canvas.onpointermove = (e) => {
    if (from == null) return;
    const to = at(e);
    if ((Math.abs(to - from) / hit.duration) * canvas.clientWidth < 4) return;   // a click that wobbled
    dragged = true;
    gen.span = hit.part = [Math.min(from, to), Math.max(from, to)];
    draw(item);
  };
  canvas.onpointerup = () => { from = null; };
  canvas.onpointercancel = () => { from = null; };
  canvas.onclick = (e) => {
    if (dragged) { dragged = false; return; }   // the click that ends a drag
    if (gen.make === 'longer') { gen.marker = hit.marker = Math.max(1, at(e)); draw(item); return; }
    play(e);
  };
}
```

- [ ] **Step 3: Look at it (no model needed)**

On the scratch server:
- Redo a part: dragging on the sample's waveform draws a band that follows the pointer; dragging again replaces it; a plain click still plays from that point.
- Longer: a line sits at the end; clicking moves it and shades what will be replaced.
- Changing Make away and back keeps the band and the line; opening another file's generate view starts with none.
- Resizing the window redraws the band in the right place.

- [ ] **Step 4: Commit**

```bash
git add src/audio_embed/web/index.html
git commit -m "Mark the part to redo, and where new sound takes over, on the waveform"
```

---

### Task 11: Work from this

**Files:**
- Modify: `src/audio_embed/web/index.html` (`row` at 1070, `showGenerate` at 795, `keepAddress` at 370, the address reader at 916)

**Interfaces:**
- Consumes: `GET /api/generated?...&clip=ID` giving `working_from` and `gone` (Task 8); `working`, `base` (Task 9).
- Produces: `query.clip` (a clip id or undefined) on a `generate` query, kept in the address as `clip=`; `make=` in the address.

- [ ] **Step 1: Put the button on every clip row**

In `row`, replace the `const like = ...` line with:

```js
  // A clip that has not been kept is not in the index, so there is nothing to find similar to it yet.
  // It can be worked from instead: looped, made longer, varied again.
  const like = hit.clip
    ? Object.assign(plain('Work from this', () => ask({ ...query, clip: hit.clip })), { className: 'plain like', title: 'Make this clip the sample: redo a part of it, loop it, make it longer or vary it again' })
    : el('button', { className: 'plain like', type: 'button' }, 'Find similar');
```

- [ ] **Step 2: Ask for the clip, and say which clip the view works from**

In `showGenerate`, replace the `fetch(fromText ? '/api/generated?text=1' : ...)` URL with:

```js
  fetch(`/api/generated?${fromText ? 'text=1' : `from=${query.id}`}${query.clip ? `&clip=${query.clip}` : ''}`)
```

After the line `if (noSample) gen.make = 'variations';` add:

```js
    const origin = el('p', { className: 'note wide', hidden: !working && !body.gone });
    if (working) origin.append(`Working from ${base.name}, a clip that has not been kept. `, plain(fromText ? 'Back to the text' : 'Back to the file', () => ask({ ...query, clip: undefined })));
    else if (body.gone) origin.textContent = 'That clip is no longer in the cache, so this is what it was made from.';
```

and change the three lines that fill the column to:

```js
    if (noSample) column.append(origin, panel, clips);
    else if (!working && body.source.audio === 'none') {
      column.append(sample, el('p', { className: 'note wide' }, 'The original is online-only and its render is not within reach, so there is nothing to generate from. Connect the render drive, or download the file.'));
      return;
    }
    else column.append(origin, sample, panel, clips);
```

- [ ] **Step 3: Keep the clip and the choice in the address**

In `keepAddress`, after the two `generate` lines, add:

```js
  if (query && query.kind === 'generate' && query.clip) params.set('clip', query.clip);
  if (query && query.kind === 'generate' && gen.make !== 'variations') params.set('make', gen.make);
```

In the address reader, replace the two `generate` lines with:

```js
  else if (params.get('generate')) {
    if (MAKES.some(([value]) => value === params.get('make'))) gen.make = params.get('make');
    const clip = /^[0-9a-f]{16}$/.test(params.get('clip') || '') ? params.get('clip') : undefined;
    ask(params.get('generate') === 'text'
      ? { kind: 'generate', text: params.get('prompt') || '', clip }
      : { kind: 'generate', id: Number(params.get('generate')), name: params.get('name') || 'this file', start: Number(params.get('at')) || 0, clip });
  }
```

- [ ] **Step 4: Look at it**

On the scratch server, with clips already in its cache if there are any (making new ones needs the model and the user's go-ahead):
- Each clip row has Work from this where a library row has Find similar.
- Pressing it shows the clip as the sample, with "Working from ..." and a way back, and the address gains `clip=`.
- Reloading the page comes back to the same clip and the same Make choice.
- Putting a made-up id in the address (`clip=0000000000000000`) shows "That clip is no longer in the cache" and the file's own view.

- [ ] **Step 5: Commit**

```bash
git add src/audio_embed/web/index.html
git commit -m "Work from a clip in the generate view before anything is kept"
```

---

### Task 12: Say what is new, and try it end to end (needs the user's go-ahead; runs Stable Audio 3)

**Files:**
- Modify: `README.md` (the section on generating)
- Modify: `CLAUDE.md` (Architecture: add `splice.py`; "Things that are not obvious")
- Modify: `docs/superpowers/specs/2026-10-08-inpainting-design.md` (only if Task 2 dropped a kind or changed the fade)

- [ ] **Step 1: README**

In the part of `README.md` that describes Generate from this, add after the paragraph on How far:

```markdown
**Make** chooses what a run does with the sample. *Variations* is the above. *Redo a part*:
drag on the sample's waveform and only that part is made anew. *Loop*: the end and the start
are made anew so the clip repeats without a seam; Join says how much. *Longer*: new sound
follows on from the sample; click its waveform to start earlier than the end, before a fade.
In all three, everything outside the new part is the sample itself.

Every clip row has **Work from this**, which makes that clip the sample: loop a variation or
make a clip from text longer before keeping anything.
```

- [ ] **Step 2: CLAUDE.md**

In the Architecture list, after the `generate.py` entry, add:

```markdown
- `splice.py`: sample arithmetic for clips where only a stretch is made anew (a redone part, a loop's join, new sound after the end). Stable Audio 3 keeps the rest of a clip exact only in its compressed form, so only the new stretch is taken from what it returns and joined into the passage that was sent, with 30 ms crossfades. A loop is turned so its ends meet in the middle, joined there, and turned back.
```

In "Things that are not obvious", add:

```markdown
- Stable Audio 3 counts in steps of 4096 samples and rounds `--seconds` up to a step. The passage for a loop or a longer clip is trimmed to whole steps before it is sent. `splice.clock` writes times a millionth of a second short for the same reason.
- A clip's story keeps `seconds` as the length of the passage it was cut from and `length` as the clip's own; they differ for a longer clip. A story without `make` is a variation.
```

- [ ] **Step 3: The whole path by hand, on a scratch index and a scratch kept folder**

Set up as `CLAUDE.md` says under "To test generating end to end". Then, with the user's go-ahead to run the model:

- Redo a part of a short file, of the tail of a short file, and of the middle of a file longer than a minute. Listen at both edges.
- Loop a 10-second file with each Join; the row plays on repeat. Listen to the join in Chrome and in Safari, and check the browser itself leaves no gap when repeating.
- Make a file 30 seconds longer from its end, and again with the marker pulled back before its fade.
- Work from a variation: loop it, then keep the loop. Check the kept file's comment tells the chain, and that it is searchable.
- Ask for a part with nothing marked, a loop of a 1-second file, and a part covering almost all of a file: each says why not.
- Run `uv run pytest -q`: all pass.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-10-08-inpainting-design.md
git commit -m "Describe redoing a part, looping and making longer"
```

(`CLAUDE.md` is kept on this machine only and is not committed.)
