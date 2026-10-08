# Adapters per kind of sound: the trial

> **For agentic workers:** this is a trial, not a build. Nothing here goes into `src/`. Every script lives in a scratch folder and is thrown away. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out, on this Mac and by ear, whether an adapter trained on one kind of the user's sounds is worth building commands and a panel row for.

**Why a trial first:** the spec's "Known and not known" lists four things nobody has measured: how much audio each kind has, how fast training runs here, whether an adapter is audible, and whether the character words are right. Each can end the idea or change its shape, and each takes one evening to find out.

**Spec:** `docs/superpowers/specs/2026-10-08-adapters-design.md` (written, **not yet reviewed by the user**). The build plan is written after this trial and after that review.

## Constraints

- **Every step except Step 0 runs a model and makes the Mac work. Do none of them until the user says it may.** Training (Part C) runs for an hour or more with the fans on.
- The library is read only through ffmpeg (`generate.cut`). Stable Audio 3's own reader is never pointed at library files, only at copies in the scratch folder.
- Nothing is written to `data/`, the library, the kept folder or the render drive. The index is read from a backup copy.
- The Mac has 28 GB free (2026-10-08). This trial needs about 1 GB for training weights, up to 2 GB for WAV copies (deleted after encoding) and under 1 GB for checkpoints. Check `df -h ~` first.

## What the user decides at the end

| Question | Answered by |
|---|---|
| Which kinds have enough of my own audio (10 minutes at least, 30 or more to do well)? | Part A |
| Are the character words right often enough to caption with? | Part B |
| How long does training take here? | Part C |
| Does the Drone adapter make drones more like mine, and turn a texture into a drone? | Part D |

---

### Step 0: Set up (no model)

- [ ] Make the scratch folder and a backup of the index:

```bash
export SCRATCH=$(mktemp -d /tmp/adapters-trial.XXXX)
cd /Users/isidor/git/audio_embed
mkdir -p "$SCRATCH/data"
sqlite3 data/index.db ".backup '$SCRATCH/data/index.db'"
cp data/locations.json "$SCRATCH/data/" 2>/dev/null; cp data/categories.json "$SCRATCH/data/" 2>/dev/null
df -h ~ | tail -1
echo "$SCRATCH"
```

Expected: a path is printed and at least 5 GB is free.

---

### Part A: How much audio each kind has

- [ ] **A1. Save this as `$SCRATCH/kinds.py`:**

```python
"""Count the files and minutes of each kind of sound, by who made them. Throwaway."""
import sys
from collections import defaultdict
from pathlib import Path

from audio_embed import audio, classify, search
from audio_embed.cli import load_embedder
from audio_embed.saved import Collections
from audio_embed.store import Store

data = Path(sys.argv[1])
OWN = {"own recording", "own show work"}
matrix = Store(data / "index.db").load(classify.MODEL)
categories = classify.load_categories(data / "categories.json")
means = search.file_means(matrix)
best = classify.assign(means, load_embedder(classify.MODEL).embed_text([d for _, d in categories]))

minutes, files = defaultdict(float), defaultdict(int)
collections = Collections(data / "collections")
for path, kind, duration, labels in zip(matrix.paths, best, matrix.durations, matrix.labels):
    name = categories[kind][0]
    source = (labels.get("source") or ["unlabelled"])[0]
    who = "own" if source in OWN else source
    files[name, who] += 1
    minutes[name, who] += duration / 60
    if who == "own" and audio.is_local(path):
        collections.add(f"Training {name}", Path(path))

sources = sorted({who for _, who in files})
print(f"{'kind':24}" + "".join(f"{s:>22}" for s in sources))
for name, _ in categories:
    print(f"{name:24}" + "".join(f"{files[name, s]:>6} files {minutes[name, s]:>7.0f} min" for s in sources))
```

- [ ] **A2. Run it** (loads CLAP for a few seconds):

```bash
uv run python "$SCRATCH/kinds.py" "$SCRATCH/data"
```

Expected: a table with one row per kind and a column per source, and a scratch collection `Training <kind>` for every kind that has files of the user's own. The collections are folders of shortcuts in `$SCRATCH/data/collections/`, not in the real `data/`.

- [ ] **A3. Show the user the table.** Kinds with under 10 minutes of their own audio cannot be trained from their own files alone. Agree on the kind for the trial; this plan assumes **Drone**. If Drone has too little, pick the kind with the most, or ask whether sample packs may be added for that kind.

- [ ] **A4. The user prunes the trial set.** Start a second server on the scratch index and let the user play the `Training Drone` collection and take out what is not a drone:

```bash
uv run audio-embed --db "$SCRATCH/data/index.db" serve --port 8766 --no-open
```

Open `http://127.0.0.1:8766`, choose the collection, remove the wrong files. Note how many were wrong: it says how far the kinds can be trusted without a person.

---

### Part B: Are the character words right?

- [ ] **B1. Save this as `$SCRATCH/character.py`:**

```python
"""Put the clearest files on each side of each pair into scratch collections, for listening. Throwaway."""
import sys
from pathlib import Path

import numpy as np

from audio_embed import classify, search
from audio_embed.cli import load_embedder
from audio_embed.saved import Collections
from audio_embed.store import Store

PAIRS = [
    ("spacy", "a distant sound with long reverb in a large space", "close", "a dry, close-up sound right at the microphone"),
    ("intense", "an intense, loud, aggressive sound", "soft", "a soft, gentle, quiet sound"),
    ("dark", "a dark, low, muffled sound", "bright", "a bright, shimmering, high-pitched sound"),
]
data = Path(sys.argv[1])
matrix = Store(data / "index.db").load(classify.MODEL)
means = search.file_means(matrix)
embedder = load_embedder(classify.MODEL)
collections = Collections(data / "collections")
for one, one_says, other, other_says in PAIRS:
    a, b = embedder.embed_text([one_says, other_says])
    lean = means @ a - means @ b   # above zero: nearer the first word
    order = np.argsort(lean)
    for word, picks in ((one, order[::-1][:20]), (other, order[:20]), (f"between {one} and {other}", order[len(order) // 2 - 10: len(order) // 2 + 10])):
        for i in picks:
            collections.add(f"Character {word}", Path(matrix.paths[i]))
    print(f"{one}/{other}: lean from {lean.min():+.3f} to {lean.max():+.3f}, middle half within {np.percentile(lean, 25):+.3f} and {np.percentile(lean, 75):+.3f}")
```

- [ ] **B2. Run it, then reload the scratch page:**

```bash
uv run python "$SCRATCH/character.py" "$SCRATCH/data"
```

Expected: three lines of numbers, and nine new collections on the scratch page: the 20 clearest files for each of the six words, and 20 from the middle of each pair.

- [ ] **B3. The user listens** to each `Character ...` collection and says, per word, roughly how many of the 20 fit. Write the counts down here. Under about 15 of 20 for a word means its description needs rewording (edit `PAIRS`, delete that collection's folder, run again) or the word is dropped. The "between" collections show what an unclear file sounds like, for deciding where "clearly on one side" should be drawn.

---

### Part C: Train one adapter

- [ ] **C1. Save this as `$SCRATCH/prepare.py`:**

```python
"""Copy a scratch collection as WAVs with captions, for Stable Audio 3's trainer. Throwaway."""
import json
import sys
from pathlib import Path

from audio_embed import audio, generate
from audio_embed.saved import Collections
from audio_embed.store import Store

data, kind = Path(sys.argv[1]), sys.argv[2]
word = kind.lower()
out = data.parent / "set"
out.mkdir(exist_ok=True)
durations = dict(zip(*(lambda m: (m.paths, m.durations))(Store(data / "index.db").load("clap"))))
made, minutes, skipped = 0, 0.0, 0
for n, path in enumerate(Collections(data / "collections").paths(f"Training {kind}")):
    if not audio.is_local(path) or path not in durations:
        skipped += 1
        continue
    seconds = min(durations[path], 600.0)
    generate.cut(Path(path), 0.0, seconds, out / f"{n:04d}.wav")   # the library is read through ffmpeg only
    (out / f"{n:04d}.txt").write_text("")                           # no character words in the trial
    made, minutes = made + 1, minutes + seconds / 60
(data.parent / "prompt.json").write_text(json.dumps({
    "use_tags": True, "tag_keys": ["prompt"], "hide_tag_names": True, "split_commas": True, "shuffle": True,
    "trigger": word, "trigger_pct": 80,
}))
(data.parent / "demos.json").write_text(json.dumps([
    {"prompt": word, "seed": 1, "duration": 20},
    {"prompt": f"{word}, dark", "seed": 2, "duration": 20},
    {"prompt": "rain on a tin roof", "seed": 3, "duration": 20},
]))
print(f"{made} files, {minutes:.0f} minutes, {skipped} skipped (online-only or not indexed by CLAP)")
```

The third demo prompt has nothing to do with drones on purpose: if it starts to sound like a drone, the adapter has taken over more than its word.

- [ ] **C2. Make the set and encode it:**

```bash
uv run python "$SCRATCH/prepare.py" "$SCRATCH/data" Drone
du -sh "$SCRATCH/set"
cd /Users/isidor/git/stable-audio-3/optimized/mlx
.venv/bin/python scripts/pre_encode_mlx.py --audio-dir "$SCRATCH/set" --output-dir "$SCRATCH/latents" --codec same-s
ls "$SCRATCH/latents" | head; ls "$SCRATCH/latents"/*.npy | wc -l
rm -r "$SCRATCH/set"
```

Expected: a line with the number of files and minutes; one `.npy` and one `.json` per file in `latents`. If `pre_encode_mlx.py` stops on a file, note its name and remove that WAV and caption, then run it again (it skips what is already encoded).

- [ ] **C3. Train for 2,000 steps on the small sfx model, and time it:**

```bash
cd /Users/isidor/git/stable-audio-3/optimized/mlx
time caffeinate -i .venv/bin/python scripts/lora_train_mlx.py \
    --dit sm-sfx --latents-dir "$SCRATCH/latents" --lr 1e-4 --name drone-trial \
    --save-dir "$SCRATCH/runs" --prompt-config "$SCRATCH/prompt.json" \
    --max-steps 2000 --checkpoint-every 500 \
    --demo-every 500 --demo-config "$SCRATCH/demos.json" --demo-dir "$SCRATCH/demos"
```

Expected: on first use it downloads `dit_sm-sfx-base_f16.npz` (0.9 GB) from Hugging Face; then a progress line per step with its speed. Write down the steps per second after the first hundred steps, and the wall time. Checkpoints appear in `$SCRATCH/runs/drone-trial/<id>/checkpoints/` and demo MP3s in `$SCRATCH/demos/` at steps 0, 500, 1000, 1500 and 2000. If memory runs short, add `--latent-crop-length 512` (about 48 seconds per excerpt) and say so in the notes.

Stable Audio 3's makers put a good adapter "around 10k" steps. 2,000 is enough to hear whether anything happens, and the measured speed says how long 10,000 would take here.

---

### Part D: Listen

- [ ] **D1. Make pairs that differ only in the adapter:**

```bash
cd /Users/isidor/git/stable-audio-3/optimized/mlx
CKPT=$(ls "$SCRATCH"/runs/drone-trial/*/checkpoints/*step=2000*.safetensors)
mkdir -p "$SCRATCH/compare"
for seed in 1 2 3; do
  .venv/bin/python scripts/sa3_mlx.py --prompt "drone, dark" --dit sm-sfx --decoder same-s --seconds 20 --seed $seed --out "$SCRATCH/compare/text-$seed-plain.wav"
  .venv/bin/python scripts/sa3_mlx.py --prompt "drone, dark" --dit sm-sfx --decoder same-s --seconds 20 --seed $seed --lora "$CKPT" --out "$SCRATCH/compare/text-$seed-adapter.wav"
done
```

- [ ] **D2. A texture turned into a drone** (the user's second use). Pick one texture from the library with the user:

```bash
cd /Users/isidor/git/audio_embed
uv run python -c "from pathlib import Path; import sys; from audio_embed import generate; generate.cut(Path(sys.argv[1]), 0.0, 20.0, Path(sys.argv[2]))" "/path/to/a texture.wav" "$SCRATCH/texture.wav"
cd /Users/isidor/git/stable-audio-3/optimized/mlx
for level in 0.6 0.8; do
  .venv/bin/python scripts/sa3_mlx.py --prompt "drone" --dit sm-sfx --decoder same-s --seconds 20 --seed 1 --init-audio "$SCRATCH/texture.wav" --init-noise-level $level --out "$SCRATCH/compare/texture-$level-plain.wav"
  .venv/bin/python scripts/sa3_mlx.py --prompt "drone" --dit sm-sfx --decoder same-s --seconds 20 --seed 1 --init-audio "$SCRATCH/texture.wav" --init-noise-level $level --lora "$CKPT" --out "$SCRATCH/compare/texture-$level-adapter.wav"
done
open "$SCRATCH/compare" "$SCRATCH/demos"
```

- [ ] **D3. The user listens** to each plain and adapter pair, and to the demos in step order, and answers:
  1. Does the adapter make the text clips more like the drones in the set?
  2. Does the texture come out as a drone with the adapter, and less so without?
  3. Did "rain on a tin roof" stay rain in the demos?
  4. Do the demos keep getting better to step 2,000, or was an earlier step best?

- [ ] **D4. Write the answers and the measured speed into the spec's "Known and not known" section**, then decide with the user:
  - **Worth building:** review the spec with what was learned, then write the build plan. If the speed makes 10,000 steps an overnight job, say so in the `train` command's opening line.
  - **Promising but thin:** train to 10,000 steps overnight (`--max-steps 10000`, and add `--lora-ckpt-path "$CKPT"` to carry on from step 2,000) and listen again.
  - **Not worth it:** stop here. The character words (Part B) can still become a filter on the page on their own.

- [ ] **D5. Clean up:** stop the scratch server and `rm -r "$SCRATCH"`. The training weights stay in Stable Audio 3's own cache (0.9 GB) for a later run.
