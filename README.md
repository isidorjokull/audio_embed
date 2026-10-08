# audio-embed

Search your own audio library by describing what you want to hear.

Type "a dark, ominous low drone" and get the files that sound like it, with the matching
passage marked on the waveform. Pick any result and ask for more files that sound like that
one. Everything runs on your own machine: your audio is never uploaded anywhere.

It was built for a sound designer's working library (sound effects, foley, field recordings,
sample packs, show bounces), where file names say little and there are far too many files to
audition by hand.

## What it does

- **Search by description.** Free text, matched against the sound itself rather than the file name.
- **Search by example.** "More like this one", from any result.
- **Long files match on a passage.** A 40-minute recording is found by its best 10 seconds, and
  playback starts there.
- **Filters.** Length, tempo, key, loop or one-shot, category, pack, and anything else read from
  file names, folders and embedded tags. A sound category ("drone", "impact", "texture"...) is
  assigned to every file automatically.
- **Votes that steer.** Thumbs up or down on results; a search can then be re-ranked toward what
  you liked and away from what you rejected. Votes on similarly worded searches count too, so a
  new wording starts from what you already taught it.
- **Collections.** Keep files in named collections. Each one is a plain folder of shortcuts you
  can open in Finder or add to your DAW's browser.
- **Mood presets.** Save a description under a short name and run it with one click.
- **Duplicate finder.** Byte-for-byte identical files, with the space they waste.
- **Renders.** A small listening copy of every file on an external drive, so you can still
  audition files whose originals are online-only in a cloud-synced folder.

The audio library itself is only ever read. Nothing is moved, renamed, changed or deleted.

## Requirements

- macOS on Apple Silicon is what it is built and used on. The models also run on CUDA or CPU,
  but other platforms are untested, and "Show in Finder" is macOS-only.
- [uv](https://docs.astral.sh/uv/) to manage Python and the dependencies (Python 3.12 is
  installed for you).
- [ffmpeg](https://ffmpeg.org) and `ffprobe` on your `PATH`. All audio is decoded through them.

```bash
brew install uv ffmpeg
```

## Installation

```bash
git clone https://github.com/isidorjokull/audio_embed.git
cd audio_embed
uv sync
```

The first time you index or search, the two models are downloaded from Hugging Face (a few
gigabytes together) and cached. After that no network connection is needed.

## Getting started

**1. Index a folder.** Start with something small to see how it behaves.

```bash
uv run audio-embed index ~/Music/Samples --model clap
```

This listens to every audio file under the folder and stores a compact description of each
one in `data/index.db`. Run it again whenever you like: only new or changed files are read.
Leaving out `--model clap` indexes with both models, which takes three to four times as long
(see [The two models](#the-two-models)).

**2. Read the labels.** This fills the filters from file names, folders and embedded tags.

```bash
uv run audio-embed labels
```

**3. Open the page.**

```bash
uv run audio-embed serve
```

Your browser opens at `http://127.0.0.1:8765`. Describe a sound and press Return.

| Key | Does |
|---|---|
| `/` | Jump to the search box |
| `↑` `↓` | Audition the previous or next result |
| `space` | Play or pause |
| `S` | Find files similar to the current one |
| `G` / `B` | Mark the current result a good or bad match |
| `K` | Keep the current file in the chosen collection |
| `R` | Rank again using your votes |

You can keep indexing more folders while the page is open; new files appear within seconds.

### From the command line

```bash
uv run audio-embed search "rain on a tin roof" -k 20
uv run audio-embed similar path/to/a/file.wav
```

## All commands

| Command | Does |
|---|---|
| `index FOLDER [--model clap] [--tag LABEL]` | Index new or changed files under a folder. `--tag` marks them all with a label of yours. |
| `labels` | Rebuild the filter labels of every indexed file. |
| `outlines` | Build the waveform of any indexed file that has none yet. |
| `serve [--port 8765] [--no-open]` | Run the browser page. |
| `search "text" [-k 10]` | Search from the terminal. |
| `similar FILE [-k 10]` | Find files that sound like a file. |
| `duplicates [--out FILE]` | Write every set of identical files to `data/duplicates.csv`. |
| `renders [--to FOLDER]` | Make listening copies on a drive (see [Renders](#renders)). |

Every command accepts `--db PATH` before its name to use a different index, for example
`uv run audio-embed --db /tmp/try/index.db index SomeFolder`.

## The two models

| | CLAP | EmbeddingGemma 2 |
|---|---|---|
| Listens in windows of | 10 seconds | 30 seconds |
| Good at | Sound effects, textures, music samples | Long recordings, speech, environments |
| Speed | Faster | Slower |

The page can show either, or both side by side. CLAP is the better starting point for
sound-design material. Scores are only comparable within one model.

## Telling it about your folders

Create `data/locations.json` to label everything under a folder. `{0}` stands for the first
folder below the one named, `{1}` for the next; a deeper folder overrides a shallower one.

```json
{
  "/Users/you/Samples/Packs": { "source": "sample pack", "pack": "{0}" },
  "/Users/you/Recordings": { "source": "own recording" },
  "/Users/you/Shows": { "source": "own show work", "project": "{0}" }
}
```

Run `uv run audio-embed labels` afterwards. Each kind (`source`, `pack`, `project`, or any
name you invent) becomes a filter on the page.

To change the automatic sound categories, put your own list in `data/categories.json`:

```json
[{ "name": "Drone", "description": "a sustained low drone" }]
```

## Renders

If your library lives in a cloud-synced folder and your disk is full, files end up
online-only: they stay in search results but cannot be played. Renders solve that.

```bash
uv run audio-embed renders --to "/Volumes/MyDrive/audio-embed renders"   # the first time
uv run audio-embed renders                                               # afterwards
```

- Every indexed file gets one listening copy on the drive: AAC at 192 kbps, or an exact copy of
  the original where that would be smaller. A render is never larger than its original.
- Identical files share one render.
- The page plays the original when it is on your machine, the render when it is not, and lists
  the file as "online only" when the drive is unplugged. Search works either way.
- `index` renders the folder it has just scanned, so the routine for an online-only folder is:
  make it available offline, index it, set it back to online-only.

As a guide, 34,000 files (100 hours of audio) gave a 190 MB index and 8 GB of renders.

## Where things are kept

Everything the tool writes goes into `data/` next to the code (not tracked by git), plus the
render folder if you set one up.

| File | What it is | If you delete it |
|---|---|---|
| `index.db` | The index | Re-index to rebuild |
| `outlines.db` | Stored waveforms | Rebuilt as needed |
| `previews/` | Converted copies of files a browser cannot play | Rebuilt as needed |
| `feedback.jsonl` | Your votes | Gone |
| `collections/` | Your collections, as folders of shortcuts | Gone |
| `moods.json`, `locations.json`, `categories.json` | Your presets and labelling | Gone |
| `settings.json`, `renders.db` | Where renders live, and which render belongs to which file | A copy of `renders.db` is kept in the render folder |

## Privacy

- Indexing, search and playback all happen locally. The only network use is the one-time model
  download.
- The page is served on `127.0.0.1` only, so it is not reachable from other machines.
- Audio is served by index id, and only for files you have indexed.

## Development

```bash
uv run pytest -q
```

The tests are fast and load no models. The code is in `src/audio_embed/`, one concern per
module, with the whole page in `web/index.html`.

## Licence

MIT. See [LICENSE](LICENSE).
