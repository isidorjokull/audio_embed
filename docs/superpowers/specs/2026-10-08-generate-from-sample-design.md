# Generate from a sample

Date: 2026-10-08. The choices marked "user" were made by the person the tool was built for.

## Why

A search finds what is already in the library. Often the right sound is close to a file but
not it: the same pad with more shimmer, the same chimes but heavier. The user has Stable
Audio 3 on the same Mac, which makes a variation of a clip from the clip, a text prompt and a
"how far to drift" amount in about six seconds. A test on three library files showed the
medium model keeps a clip recognisable while moving it toward the prompt.

## What it does

The player gets a **Generate from this** button (and the `N` key). It opens a view for the file
that is loaded, with a panel (user: "simple, with a More fold"):

- **Prompt**: text added on top of the sample. May be empty.
- **How far**: Close, Medium or Far from the original (noise levels 0.4, 0.6, 0.8).
- **More**, closed by default: **Avoid** (a second prompt for what to steer away from, user),
  **Count** (1, 2, 4, 8; default 4), **Length**, **Model** (medium by default, user; the two
  small models are offered when their weights are there) and **Prompt strength**.

Pressing Generate makes that many variations, each from another seed. They appear one by one
under the panel with waveform, playback and a Keep button.

- **Only what is kept stays** (user). A new clip is temporary. Keep copies it into the library
  and makes it searchable; the rest are cleared out by age.
- A clip is as long as the sample, up to 60 s. From a longer file it is 60 s starting at the
  passage the search matched. Length under More overrides this, up to 120 s.
- Avoid only has an effect with prompt strength above 1, so a run with Avoid filled in and
  strength left at 1 uses 3.
- A sample whose original is online-only is read from its render when the drive is connected.
  Otherwise the view says so and offers nothing.
- If Stable Audio 3 or the chosen model's weights are missing, the view says what is missing.

## From text alone (added the same day, user)

**Generate from text**, in the row with the model choice and the file count, opens the same
view with no sample: whatever is in the search box becomes the prompt. There is nothing to be
near or far from, so How far is gone, and Length (10 s by default) and Model are shown up front
because they now decide the most. Clips are listed, played and kept as above; a kept one is
named by its prompt alone and described as "Generated from text".

## Where things are written

Kept clips go to **one folder inside the sample library** (user), by default
`Samples/Generated`. This is a deliberate exception to "the library is read-only", and it is
fenced the way the render folder is:

- The tool writes there only when the folder carries its marker, `.audio-embed-generated`,
  made once by `audio-embed generator --keep-in FOLDER`.
- It only ever adds new files. It never overwrites, renames or deletes one.
- Everything else in the library stays read-only.

A kept clip is named `<sample name> - <prompt> <n>.wav` and carries where it came from in its
comment tag (sample, passage, prompts, distance, model, seed), which the label reader already
shows as the file's description. The folder is given `"source": "generated"` in
`data/locations.json`, a fifth value next to the four that say who made a file.

Clips that are not kept live in `data/generating/`, each with a small JSON file saying what
it was made from. The folder is a cache: the oldest clips go once it passes 500 MB, and it can
be deleted at any time.

## Parts

- `generate.py`: everything about making a clip. Cutting the reference with ffmpeg, the
  Stable Audio 3 command line, the queue (one clip at a time, in a worker thread), the cache
  and its trimming, and keeping. Stable Audio 3 runs as a separate program from the user's
  own checkout, in that checkout's Python, so none of its dependencies enter this project.
  The checkout and the kept folder are named in `data/settings.json`.
- `server.py`: the routes. `POST /api/generate` starts a run, `GET /api/generate/{run}` says
  how far it is, `GET /api/generated?from=ID` lists the clips made from a file that are still
  in the cache, `GET /generated/{clip}` and `/api/generated/{clip}/peaks` play and draw one,
  `POST /api/generated/{clip}/keep` keeps it.
- `Library.take_in(folder)`: indexes what is new in one folder with the models the server
  already has loaded, labels it and makes its render, then has the index re-read at once.
- `cli.py`: `audio-embed generator --sa3 FOLDER --keep-in FOLDER`, the one-time setup.
- `web/index.html`: the button, the view and its rows.

## Two rules this changes

1. **Clips are served by clip id.** Until now the server only served indexed files, by file
   id. A clip id is checked to be the id of a file in the cache folder and nothing else; no
   path ever comes from the page.
2. **The server writes to the index when a clip is kept.** Until now nothing the server did
   touched `index.db`, so the page never had to re-read it while someone browsed. A keep is
   rare and deliberate, and re-reading is exactly what should follow it. Nothing else the
   server does writes there.

## Not in this version

- Generating from several files at once.
- Deleting a kept clip from the page. It is a file in the library: remove it in Finder, and
  the next run of the indexer on that folder drops it from the index.
- Choosing the best clips automatically by their closeness to the sample and the prompt.

## Testing

Unit tests cover the pure parts (which passage is used, the command line, file names, the
description, request checking) and, with a stand-in for Stable Audio 3 that copies the
reference, the whole path from a request to a kept, indexed and labelled file, including the
refusal to write into a folder without the marker. The page and the real model are checked
by hand against a scratch index and a scratch kept folder.
