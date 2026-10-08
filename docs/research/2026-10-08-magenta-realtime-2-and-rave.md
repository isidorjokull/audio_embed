# Magenta RealTime 2 and RAVE: findings

Researched 2026-10-08 for a MacBook Pro, M2 Max, 32 GB. Nothing was installed or run; every
finding comes from reading repositories, documentation, licence files and issue trackers.

**Decision (user, 2026-10-08):** Magenta RealTime 2 sounds good, but belongs in a separate
toolkit, not in audio-embed. RAVE is not pursued. This file is the starting point for that
separate toolkit.

How sure each finding is:

- **Verified**: read at the primary source.
- **Inferred**: concluded from reading source code, not run.
- **Secondary**: from a third party.
- **Not found**: looked for and not established.

---

## Magenta RealTime 2

### In short

- Makes music continuously from text, audio examples and MIDI notes. Trained on about 71,000
  hours of mostly instrumental stock music (verified, model card).
- Licence is clean for paid work: code Apache-2.0, weights CC BY 4.0, and "Google claims no
  rights in outputs you generate" (verified: `MODEL.md`, `LICENSE`, Hugging Face card).
- **The deciding unknown:** whether it makes usable drones, ambiences and textures, and
  whether a non-musical audio reference steers it anywhere useful. No test of this was found.
  Musical pads are plausible (Google's own example prompt is "ambient pads with sub bass");
  foley and field recordings are probably outside what it knows (inferred).
- An audio reference steers **style**. It is read as a 10-second, 16 kHz mono summary of the
  file. It does not rebuild the file the way Stable Audio 3's `--init-audio` does, so it is
  not a "variation" in audio-embed's sense.

### Try it first, at no cost in code

1. Install the prebuilt AU plugin or the standalone app. Both take audio prompts ("Type a
   prompt or upload a sample") and MIDI. Load a drone, a texture, a field recording and a
   musical loop and listen. This answers the deciding unknown.
2. `mrt mlx generate` with `mrt2_base` and `mrt2_small` for 30 seconds; read the printed steps
   per second.
3. Run the same prompt twice and compare the files (see "Seed" below).
4. Time a cold start.
5. Try references of 1 to 3 seconds, padded with silence against looped to 10 seconds.

### Install (verified)

```bash
# Python 3.12 only: 3.13 with mlx 0.32.3 fails to load the small model (issue #103, open)
uv pip install "magenta-rt[mlx]"
mrt models init
mrt models download
```

- Needs macOS 14 or later. The Hugging Face repository is not gated.
- Everything lives in `~/Documents/Magenta/magenta-rt-v2/` (`resources/`, `models/<name>/`,
  `outputs/`). `MAGENTA_HOME` moves the whole tree.
- Disk: shared resources 1.38 GB, `mrt2_base` 2.79 GB, `mrt2_small` 0.46 GB. Base alone with
  resources is about 4.2 GB. The raw checkpoints (9.8 GB and 1.1 GB) are not needed.

### Audio reference: where it exists

| Where | Audio reference | Source |
|---|---|---|
| Model | Yes | `MODEL.md` |
| `mrt mlx generate` (release 2.0.3) | **No**, text only | `magenta_rt/cli/mlx_commands.py` |
| Python library | Yes: `embed_style(text_or_audio, ...)` | `magenta_rt/mlx/system.py`, `docs/inference.md` |
| AU plugin and standalone app | Yes: prompt slots read the first 10 s of a file | `examples/mrt2/auv3/README.md` |
| AU plugin "Custom" bank | Continues from a file of up to 28 s; the join is not seamless | same |
| Max, Pd, SuperCollider externals | No, text only | `examples/max/README.md` |
| Live audio input | No, listed as future work | launch blog |

The documented Python call:

```python
from magenta_rt.audio import Waveform
wav = Waveform.from_file("jazz_piano_trio.wav")
m.tokenize(m.embed(wav))
```

Pull request #58 adds `--audio`, `--blend-ratio` and `--output` to the command line. It has
been open since 2026-06-15 with a failed CLA check and no reply from a maintainer.

Continuing from a file is not possible offline: only the C++ engine and the plugin can prefill
with real audio.

### The offline command (verified unless marked)

```bash
mrt mlx generate --prompt "disco funk" --duration 4.0 --model=mrt2_base
```

- Options: `--prompt`, `--model`, `--duration`, `--bits`, `--temperature`, `--top-k`,
  `--cfg-musiccoca`, `--cfg-notes`, `--checkpoint`, `--mlxfn/--no-mlxfn`. No audio, output or
  seed option.
- **Output path is fixed**: `~/Documents/Magenta/magenta-rt-v2/outputs/output_audio_mlx_<model>.wav`,
  overwritten each time.
- Output is 48 kHz stereo WAV; 16-bit is inferred from the library's defaults, not run.
- No maximum length. The model sees about 20 seconds of its own past, so it builds no long
  form.
- **Seed (inferred):** there is no seed option, and the random key lives in a state file the
  Python wrapper always starts from, so two runs of one prompt may be identical. Test this.
- The C++ example `hello_mrt2` has `--output`, `--force`, `--seed-rotation` and `--prompt`,
  takes text only, and must be built with CMake.

### MIDI and other controls (verified)

- Notes: a 128-pitch on/off state per frame; no velocity. Engine calls `set_note_on(n)` and
  `set_note_off(n)`; the Max external has `noteon`/`noteoff`; the standalone app makes a
  virtual MIDI port "Magenta RT Input".
- Drums can only be switched off (`drumless`), not played.
- Also: temperature, top-k, guidance for style, notes and drums, up to six weighted prompts.
- The Python `generate` command passes no notes.

### Speed on this Mac

- The docs list the M2 Max as real-time for `mrt2_base` (README, `docs/models.md` and Google's
  page agree; the README's "Pro Max" is a typo for "Pro/Max").
- Users dispute it: an M3 Max owner measured 56 ms a frame against a 40 ms budget, and an M4
  Max owner reports "very often hitting the limits" (issue #39, unresolved).
- Expect real time with the base model to be marginal here. Offline generation does not need
  real time, and `mrt2_small` runs in real time on any Apple Silicon Mac.
- Memory is not documented. The base model is 2.8 GB on disk, so 32 GB should be ample
  (inferred).

### What users say

Thin. One user: "some genres are really well represented and other common genres aren't",
and prompts stop mattering past about 15 tokens (issue #104). One user on an M4 with 32 GB:
"random granular synth like behaviour. Is this it?" (issue #55). The Hacker News thread has
no listening reports, and the blogs found restate the launch material.

### If it is ever wrapped as a generator

It fits the pattern audio-embed uses for Stable Audio 3 (a separate program, one call per
clip, a reference file in, a WAV out) with one addition: a wrapper script of about 40 lines,
run by Magenta's own Python, taking `--prompt`, `--audio`, `--seconds`, `--seed`, `--model`
and `--out`. The calls it needs exist in the library: `MagentaRT2StdMlxfn(...)`,
`Waveform.from_file(ref)`, `embed_style(...)` for the text and the audio and a weighted mix
of the two, `generate(conditioning=..., frames=int(seconds * 25))`, `wav.write(out)`.
Each call reloads a 2.8 GB model; the load time is not documented.

### Links

- Repository: https://github.com/magenta/magenta-realtime
- Model card and licence: https://github.com/magenta/magenta-realtime/blob/main/MODEL.md
- Install: https://magenta.github.io/magenta-realtime/installation.html
- Inference: https://github.com/magenta/magenta-realtime/blob/main/docs/inference.md
- Plugin: https://github.com/magenta/magenta-realtime/blob/main/examples/mrt2/auv3/README.md
- Weights: https://huggingface.co/google/magenta-realtime-2
- Launch blog: https://magenta.withgoogle.com/magenta-realtime-2
- Issues cited: #39 (real-time speed), #55, #58 (audio on the command line), #103 (Python
  3.13), #104 (genre coverage)

---

## RAVE (ruled out)

### Why

- **Licence:** the code is CC BY-NC 4.0 (verified in the repository's `LICENSE`, changed from
  MIT on 2022-02-21; the package still carries a stale MIT label). The author said in 2023
  that sounds you make are yours to use in concerts and productions, and that the clause is
  aimed at paid products built on RAVE. That is a forum statement, not licence text.
- **Training:** release 2.3.1 refuses to train on Apple Silicon. The documentation puts
  training at three or four days plus four days to three weeks on an NVIDIA GPU, and asks for
  at least an hour (elsewhere three hours) of one homogeneous source. A mixed set of
  recordings is the wrong shape.
- **Pretrained models:** speech, singing voice, percussion, darbouka, Apollo recordings,
  vintage music, classical recordings, orchestral instruments. No licence is stated for
  IRCAM's own set. A second set from the Intelligent Instruments Lab (guitar, sax, organ,
  voices, birds, water, whales) is CC BY-NC 4.0.
- **Hazard:** in 2.3.1, `rave generate` builds its output path in a way that discards the
  output folder when the input folder is an absolute path, and then writes over the source
  files (read in `scripts/generate.py`, not run). **Never point it at the library.**

### In Ableton, if ever wanted

- The RAVE VST (AU and VST, universal build) is a beta from 2024-04-02 and its repository is
  archived.
- `nn~` is a Max external, so in Live it needs Max for Live and a device you patch yourself.
  Version 1.6.0 has open reports on Apple Silicon of crashes and very high CPU; users say
  1.5.6 avoids both.
- RAVE models have a latency that "cannot be reduced" (IRCAM tutorial).

### The same group's current work: AFTER

https://github.com/acids-ircam/AFTER blends one audio source for timbre with another audio or
MIDI source for structure. Version 2 runs on Apple Silicon (`--device mps`), ships two Max
for Live devices and has pretrained models. Licence CC BY-NC 4.0; training is still heavy
("up to a week on a single RTX 4090"); the licence of its pretrained models was not found.

### Links

- Repository: https://github.com/acids-ircam/RAVE
- Licence discussion: https://github.com/acids-ircam/RAVE/discussions/195
- Pretrained models: https://acids-ircam.github.io/rave_models_download and
  https://huggingface.co/Intelligent-Instruments-Lab/rave-models
- Training guide: https://forum.ircam.fr/article/detail/training-rave-models-on-custom-data/
- nn~ issues: https://github.com/acids-ircam/nn_tilde/issues (91, 92, 95)

---

## Open questions only a test on this Mac can answer

- Does Magenta RealTime 2 make usable drones, ambiences and textures, and does a non-musical
  audio prompt steer it?
- Offline speed and peak memory of `mrt2_base`, and whether the plugin holds real time in
  Live.
- Whether repeated Python runs are identical without changing the random key.
- Model load time per process.
- How its audio style reference compares by ear with Stable Audio 3's `--init-audio` on the
  same file.
