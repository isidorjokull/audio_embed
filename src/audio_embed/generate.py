"""Variations of a library file, made by Stable Audio 3 from the file and a prompt,
and new clips made from a prompt alone.

Stable Audio 3 is run as a separate program from the user's own checkout of it, in
that checkout's Python, so none of what it needs becomes part of this project. A run
from a sample cuts the passage to work from with ffmpeg; every run makes its clips
one at a time.

A new clip is temporary: it sits in a cache folder next to the index, with a small
JSON file saying what it was made from, and the oldest go once the cache is full.
Keeping a clip copies it into one folder of the audio library, the only place in
the library this tool ever writes. Nothing is written there unless the folder
carries the marker made by `setup`, and a file there is never overwritten.
"""

import json
import queue
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio, renders, splice

MARKER = ".audio-embed-generated"
# The cache of clips that have not been kept, next to the index.
CACHE = "generating"
CACHE_BYTES = 500 * 1024**2
# A clip is as long as its sample, up to this; a longer one has to be asked for.
LONGEST_S = 60.0
LONGEST_ASKED_S = 120.0
# A clip made from text alone has no sample to take its length from.
TEXT_SECONDS = 10.0
# What such clips are filed under, where clips from a sample are filed under the sample's path.
FROM_TEXT = "text"
# How much noise the sample is buried in before the model rebuilds it: the more, the further the clip drifts.
DISTANCES = {"close": 0.4, "medium": 0.6, "far": 0.8}
# What a run makes: whole new clips, or one stretch of the sample made anew (see splice.py).
MAKES = ("variations", "part", "loop", "longer")
# How many seconds a loop's join replaces, half of it at each end of the clip.
JOINS = {"short": 1.0, "medium": 2.0, "long": 4.0}
# The shortest part that can be redone, and how much of a passage a part must leave alone.
SHORTEST_PART_S = 0.2
MIN_KEPT_S = 1.0
MIN_LOOP_S = 2.0
# What a kept clip with no prompt is called, by what was made; anything else is a variation.
UNNAMED = {"part": "redone", "loop": "loop", "longer": "longer"}
# Our name for a model -> Stable Audio 3's names for it and for the codec it needs.
MODELS = {"medium": ("medium", "same-l"), "sfx": ("sm-sfx", "same-s"), "music": ("sm-music", "same-s")}
WEIGHTS = {
    "medium": ["dit_medium_f16.npz", "same_l_encoder_f32.npz", "same_l_decoder_f32.npz", "t5gemma_f16.npz"],
    "sfx": ["dit_sm-sfx_f16.npz", "same_s_encoder_f32.npz", "same_s_decoder_f32.npz", "t5gemma_f16.npz"],
    "music": ["dit_sm-music_f16.npz", "same_s_encoder_f32.npz", "same_s_decoder_f32.npz", "t5gemma_f16.npz"],
}
# What to avoid has no effect at prompt strength 1, so asking for it raises the strength to this.
AVOID_STRENGTH = 3.0
# How many words of the prompt go into the name of a kept clip.
NAME_WORDS = 8
SETUP = "Run `audio-embed generator --sa3 FOLDER --keep-in FOLDER` once."


@dataclass(frozen=True)
class Ask:
    """What the panel asks for."""

    prompt: str = ""
    avoid: str = ""
    distance: str = "medium"
    strength: float = 1.0
    count: int = 4
    seconds: float | None = None  # None: as long as the sample, up to LONGEST_S
    model: str = "medium"
    make: str = "variations"
    span: tuple[float, float] | None = None  # the part to redo, in seconds in the sample
    marker: float | None = None  # where new sound takes over, for longer; None: the end of the sample
    join: str = "medium"
    add: float = 30.0  # seconds of new sound, for longer


def ask_from(body: dict) -> Ask:
    """A request from the page, checked. A ValueError says what is wrong with it."""

    def text(key):
        return " ".join(str(body.get(key) or "").split())

    try:
        count = int(body.get("count", 4))
        strength = float(body.get("strength", 1.0))
        seconds = None if body.get("seconds") in (None, "") else float(body["seconds"])
    except (TypeError, ValueError):
        raise ValueError("Count, length and prompt strength must be numbers.") from None
    distance, model = body.get("distance", "medium"), body.get("model", "medium")
    if distance not in DISTANCES:
        raise ValueError(f"How far must be one of: {', '.join(DISTANCES)}.")
    if model not in MODELS:
        raise ValueError(f"The model must be one of: {', '.join(MODELS)}.")
    if not 1 <= count <= 8:
        raise ValueError("Between 1 and 8 clips can be made at a time.")
    if seconds is not None and not 1 <= seconds <= LONGEST_ASKED_S:
        raise ValueError(f"A clip can be 1 to {LONGEST_ASKED_S:g} seconds long.")
    if not 0 <= strength <= 10:
        raise ValueError("Prompt strength goes from 0 to 10.")
    avoid = text("avoid")
    if avoid and strength <= 1:
        strength = AVOID_STRENGTH
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


def passage(duration: float, matched_at: float, asked: float | None = None) -> tuple[float, float]:
    """(start, length) in seconds of the part of a sample a clip is made from.

    The whole sample when it is short enough; otherwise a stretch from where the
    search matched, pulled back if it would run past the end.
    """
    seconds = min(duration, asked or LONGEST_S)
    start = 0.0 if seconds >= duration else min(matched_at, duration - seconds)
    return float(start), float(seconds)


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


def lay_out(make: str, samples: np.ndarray, shape: Plan) -> tuple[np.ndarray, int, int, int, int]:
    """(what Stable Audio 3 is given, the start and end of the new stretch in it, the samples to ask for, how far a loop was turned)."""
    if make == "part":
        n = len(samples)
        start, end, total = splice.snap(shape.part[0]), min(splice.snap(shape.part[1]), n), n
        if n - end < splice.STEP:
            # Reaching the end means all of it. Stable Audio 3 moves the end of a range to the nearest step, so the
            # range is run to the end of the last step it works on (it pads the passage with silence itself), past
            # the end of the passage: no sliver of the old tail is left after the part.
            end = total = -(-n // splice.STEP) * splice.STEP
        return samples, max(0, min(start, end - splice.STEP)), end, total, 0
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
    if make == "loop":
        return splice.turn_back(joined, by)
    if make == "part" and end >= len(given):
        # A part that reaches the end leaves the clip its length, and nothing says the new tail has died away there.
        return splice.fade_out(joined[: len(given)])
    return joined


def cut(source: Path, start: float, seconds: float, out: Path) -> None:
    """Write one passage of a file as the 44.1 kHz 16-bit stereo WAV Stable Audio 3 reads, without its tags."""
    channels = audio.channel_count(source)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(seconds), "-i", str(source), "-map", "0:a:0"]
    if channels == 1:
        cmd += ["-af", "pan=stereo|c0=c0|c1=c0"]
    elif channels > 2:
        # ffmpeg has no downmix for arbitrary layouts, so average every channel into both sides.
        mix = "+".join(f"{1 / channels:.4f}*c{c}" for c in range(channels))
        cmd += ["-af", f"pan=stereo|c0={mix}|c1={mix}"]
    cmd += ["-map_metadata", "-1", "-ar", "44100", "-c:a", "pcm_s16le", "-f", "wav", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:200]}")


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


def kept_name(stem: str, prompt: str, taken: set[str], make: str = "variations") -> str:
    """A file name for a kept clip: its sample (if it had one), the start of its prompt, and the first number not in use.

    A clip with no prompt is named by what was made.
    """
    said = " ".join(re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", prompt).split()[:NAME_WORDS]) or UNNAMED.get(make, "variation")
    start = f"{stem[:120]} - {said}" if stem else said
    taken = {name.lower() for name in taken}
    n = 1
    while f"{start} {n}.wav".lower() in taken:
        n += 1
    return f"{start} {n}.wav"


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


def setup(root: Path) -> None:
    """Make `root` the folder kept clips go to. The folder it sits in must be there."""
    if not root.parent.is_dir():
        raise RuntimeError(f"{root.parent} is not there.")
    root.mkdir(exist_ok=True)
    (root / MARKER).touch()


def saved(settings: Path) -> tuple[Path | None, Path | None]:
    """(the Stable Audio 3 folder, the folder kept clips go to), as chosen earlier."""
    known = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    return tuple(Path(known[key]) if known.get(key) else None for key in ("generator", "generated"))


def save(settings: Path, sa3: Path | None = None, keep_root: Path | None = None) -> None:
    known = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    known.update({key: str(folder) for key, folder in (("generator", sa3), ("generated", keep_root)) if folder})
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(known, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def label_folder(locations: Path, folder: Path) -> bool:
    """Say in the locations file that everything under `folder` was generated, unless it already has an entry."""
    known = json.loads(locations.read_text(encoding="utf-8")) if locations.exists() else {}
    if str(folder) in known:
        return False
    known[str(folder)] = {"source": "generated"}
    locations.write_text(json.dumps(known, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


class Generator:
    def __init__(self, cache: Path, sa3: Path | None, keep_root: Path | None, run=subprocess.run,
                 cache_bytes: int = CACHE_BYTES) -> None:
        self.cache = cache
        self.sa3 = sa3
        self.keep_root = keep_root
        self.cache_bytes = cache_bytes
        self._run = run  # what starts Stable Audio 3; tests pass a stand-in
        self._runs: dict[str, dict] = {}
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    def problem(self, model: str) -> str | None:
        """Why clips cannot be made with a model, in words for the page; None when they can."""
        if self.sa3 is None:
            return f"No Stable Audio 3 folder has been chosen. {SETUP}"
        if not (self.sa3 / ".venv/bin/python").exists() or not (self.sa3 / "scripts/sa3_mlx.py").exists():
            return f"{self.sa3} is not a Stable Audio 3 folder that has been installed. Run its install.sh, or choose another. {SETUP}"
        missing = [name for name in WEIGHTS[model] if not (self.sa3 / "models/mlx" / name).exists()]
        if missing:
            return (
                f"The {model} model is not downloaded: {', '.join(missing)} missing from {self.sa3 / 'models/mlx'}"
                " (a link to a file that has since been deleted counts as missing)."
            )
        return None

    def models(self) -> list[str]:
        """The models that are ready to use."""
        return [model for model in MODELS if self.problem(model) is None]

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

    def start_from_clip(self, clip_id: str, ask: Ask) -> str:
        """Queue a run whose sample is a clip in the cache. Its clips are filed with the clip they came from."""
        clip, story = self.clip(clip_id), self.story(clip_id)
        if clip is None or story is None:
            raise ValueError("That clip is no longer in the cache. Generate it again.")
        with wave.open(str(clip), "rb") as f:
            duration = f.getnframes() / f.getframerate()
        return self._queue_run(clip, story["key"], story["name"], plan(duration, 0.0, ask), ask, parent=describe(story))

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

    def progress(self, run_id: str) -> dict | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        return {"id": run["id"], "done": run["done"], "clips": [dict(clip) for clip in run["clips"]]}

    def _work(self) -> None:
        while True:
            run = self._queue.get()
            try:
                self._make(run)
            except Exception as e:  # the worker must outlive any one run
                for clip in run["clips"]:
                    if clip["state"] != "done":
                        clip.update(state="failed", error=str(e)[:300])
            run["done"] = True

    def _make(self, run: dict) -> None:
        ask: Ask = run["ask"]
        shape: Plan = run["plan"]
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            reference, given, part, by = None, None, None, 0
            if run["source"] is not None:
                if run["source"].parent == self.cache and not run["source"].is_file():
                    raise RuntimeError("That clip is no longer in the cache. Generate it again.")
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
                self._trim(self._in_use())

    def _story_file(self, clip_id: str) -> Path:
        return self.cache / f"{clip_id}.json"

    def _in_use(self) -> set[str]:
        """The clips of every run that is not over, and the clips such runs are made from."""
        with self._lock:
            runs = [run for run in self._runs.values() if not run["done"]]
        used = {clip["id"] for run in runs for clip in run["clips"]}
        return used | {run["source"].stem for run in runs if run["source"] is not None and run["source"].parent == self.cache}

    def _trim(self, spare: set[str]) -> None:
        """Drop the oldest clips once the cache is over its size, never one that is in `spare`."""
        total = 0
        for clip in sorted(self.cache.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
            total += clip.stat().st_size
            if total > self.cache_bytes and clip.stem not in spare:
                clip.unlink()
                self._story_file(clip.stem).unlink(missing_ok=True)

    def clip(self, clip_id: str) -> Path | None:
        """The file of a clip in the cache. The id is all that is ever taken from the page."""
        if not re.fullmatch(r"[0-9a-f]{16}", clip_id):
            return None
        path = self.cache / f"{clip_id}.wav"
        return path if path.is_file() else None

    def story(self, clip_id: str) -> dict | None:
        """What a clip in the cache was made from."""
        if self.clip(clip_id) is None:
            return None
        try:
            return json.loads(self._story_file(clip_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def clips_from(self, key: str) -> list[dict]:
        """The clips still in the cache that were made from one library file (or from text, for FROM_TEXT), oldest first."""
        found = []
        for file in self.cache.glob("*.json") if self.cache.is_dir() else []:
            story = self.story(file.stem)
            if story is not None and story["key"] == key:
                kept = story.get("kept")
                found.append({**story, "id": story["clip"], "kept": kept if kept and Path(kept).is_file() else None})
        return sorted(found, key=lambda s: (s["made"], s["n"]))

    def keep(self, clip_id: str) -> Path:
        """Copy a clip into the kept folder under a new name, with its story as its comment tag."""
        clip, story = self.clip(clip_id), self.story(clip_id)
        if clip is None or story is None:
            raise RuntimeError("That clip is no longer in the cache. Generate it again.")
        if story.get("kept") and Path(story["kept"]).is_file():
            return Path(story["kept"])
        if self.keep_root is None or not (self.keep_root / MARKER).is_file():
            where = "No folder has been chosen" if self.keep_root is None else f"{self.keep_root} is not marked as the folder"
            raise RuntimeError(f"{where} for kept clips. {SETUP}")
        with self._lock, tempfile.TemporaryDirectory() as scratch:
            tagged = Path(scratch) / "tagged.wav"
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-c", "copy", "-map_metadata", "-1",
                 "-metadata", f"comment={describe(story)}", str(tagged)],
                capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:200]}")
            target = self.keep_root / kept_name(story["name"], story["prompt"], {p.name for p in self.keep_root.iterdir()},
                                                story.get("make", "variations"))
            # Opened so that an existing file is an error: nothing in the library is ever overwritten.
            with tagged.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            story["kept"] = str(target)
            self._story_file(clip_id).write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")
        return target


def beside(db_path: Path) -> Generator:
    """The generator whose cache and settings live next to an index."""
    return Generator(db_path.parent / CACHE, *saved(db_path.parent / renders.SETTINGS))
