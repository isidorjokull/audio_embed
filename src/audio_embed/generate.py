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
from dataclasses import dataclass
from pathlib import Path

from . import audio, renders

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


def command(sa3: Path, ask: Ask, reference: Path | None, seconds: float, seed: int, out: Path) -> list[str]:
    """The Stable Audio 3 command line for one clip. Without a `reference` the clip is made from the prompt alone."""
    dit, decoder = MODELS[ask.model]
    cmd = [
        str(sa3 / ".venv/bin/python"), str(sa3 / "scripts/sa3_mlx.py"),
        "--prompt", ask.prompt, "--dit", dit, "--decoder", decoder,
    ]
    if reference is not None:
        cmd += ["--init-audio", str(reference), "--init-noise-level", str(DISTANCES[ask.distance])]
    cmd += ["--seconds", str(round(seconds, 2)), "--seed", str(seed), "--out", str(out)]
    if ask.strength != 1.0:
        cmd += ["--cfg", str(ask.strength)]
    if ask.avoid:
        cmd += ["--negative-prompt", ask.avoid]
    return cmd


def kept_name(stem: str, prompt: str, taken: set[str]) -> str:
    """A file name for a kept clip: its sample (if it had one), the start of its prompt, and the first number not in use."""
    said = " ".join(re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", prompt).split()[:NAME_WORDS]) or "variation"
    start = f"{stem[:120]} - {said}" if stem else said
    taken = {name.lower() for name in taken}
    n = 1
    while f"{start} {n}.wav".lower() in taken:
        n += 1
    return f"{start} {n}.wav"


def describe(made: dict) -> str:
    """Where a clip came from, in a sentence or two, for its comment tag."""

    def clock(s):
        return f"{int(s // 60)}:{int(s % 60):02d}"

    from_text = made.get("key") == FROM_TEXT
    whole = made.get("whole", made["start_s"] == 0)
    at = "" if whole else f" ({clock(made['start_s'])} to {clock(made['start_s'] + made['seconds'])})"
    parts = ["Generated from text." if from_text else f'Generated from "{made["name"]}"{at}.']
    if made["prompt"]:
        parts.append(f"Prompt: {made['prompt'].rstrip('.')}.")
    if made["avoid"]:
        parts.append(f"Avoid: {made['avoid'].rstrip('.')}.")
    if not from_text:
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
        """Queue a run of `ask.count` clips and give its id.

        `source` is the file to read, `key` the library path the clips belong to
        (they differ when an online-only file is read from its render) and `name`
        the sample's name without its suffix.
        """
        start, seconds = passage(duration, start_s, ask.seconds)
        return self._queue_run(Path(source), key, name, start, seconds, seconds >= duration - 0.05, ask)

    def start_from_text(self, ask: Ask) -> str:
        """Queue a run of clips made from the prompt alone, with no sample, and give its id."""
        if not ask.prompt:
            raise ValueError("Type what to generate first.")
        return self._queue_run(None, FROM_TEXT, "", 0.0, ask.seconds or TEXT_SECONDS, True, ask)

    def _queue_run(self, source: Path | None, key: str, name: str, start: float, seconds: float, whole: bool, ask: Ask) -> str:
        run = {
            "id": secrets.token_hex(8), "source": source, "key": key, "name": name,
            "start_s": start, "seconds": seconds, "whole": whole, "ask": ask, "done": False,
            "clips": [
                {"id": secrets.token_hex(8), "n": n, "state": "waiting", "seconds": seconds, "error": None}
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
        self.cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            reference = None
            if run["source"] is not None:
                reference = Path(scratch) / "reference.wav"
                cut(run["source"], run["start_s"], run["seconds"], reference)
            for clip in run["clips"]:
                clip["state"] = "making"
                seed = secrets.randbelow(2**31)
                out = Path(scratch) / f"{clip['id']}.wav"
                result = self._run(
                    command(self.sa3, ask, reference, run["seconds"], seed, out),
                    cwd=self.sa3, capture_output=True, text=True,
                )
                if result.returncode or not out.is_file():
                    said = [line for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
                    clip.update(state="failed", error=(said[-1].strip() if said else "Stable Audio 3 made no file.")[:300])
                    continue
                story = {
                    "clip": clip["id"], "n": clip["n"], "key": run["key"], "name": run["name"],
                    "start_s": run["start_s"], "seconds": run["seconds"], "whole": run["whole"],
                    "prompt": ask.prompt, "avoid": ask.avoid, "distance": ask.distance, "strength": ask.strength,
                    "model": ask.model, "seed": seed, "made": time.time(), "kept": None,
                }
                # The story first, so a clip in the cache always has one.
                self._story_file(clip["id"]).write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")
                shutil.move(out, self.cache / f"{clip['id']}.wav")
                clip["state"] = "done"
                self._trim({c["id"] for c in run["clips"]})

    def _story_file(self, clip_id: str) -> Path:
        return self.cache / f"{clip_id}.json"

    def _trim(self, spare: set[str]) -> None:
        """Drop the oldest clips once the cache is over its size, never one of the run in hand."""
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
            target = self.keep_root / kept_name(story["name"], story["prompt"], {p.name for p in self.keep_root.iterdir()})
            # Opened so that an existing file is an error: nothing in the library is ever overwritten.
            with tagged.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            story["kept"] = str(target)
            self._story_file(clip_id).write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")
        return target


def beside(db_path: Path) -> Generator:
    """The generator whose cache and settings live next to an index."""
    return Generator(db_path.parent / CACHE, *saved(db_path.parent / renders.SETTINGS))
