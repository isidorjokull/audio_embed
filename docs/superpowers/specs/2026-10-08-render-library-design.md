# Render library: listening copies on an external drive

Date: 2026-10-08. The choices marked "user" were made by the person the tool was built for.

## Why

The Mac's own disk is nearly full and most of the audio library lives in a cloud-synced folder,
where files can be made online-only. An online-only file stays in search results today, but it
is 0 bytes on disk, so it cannot be played and its waveform is not found.

The user wants to look through and listen to the whole library without the originals being on
the Mac.

## What it does

Every indexed file gets one **render**: a stereo AAC copy at 192 kbps (user), the same conversion
the page already uses for previews. Renders live in one folder on an external drive, shown
here as `/Volumes/DRIVE` (user). When a file's original is online-only, the page plays its render.

- Search needs neither the originals nor the drive. It always works.
- A file whose original is on the Mac plays from the original, as now.
- A file whose original is online-only plays from its render when the drive is connected.
- A file with neither is still listed, marked **online only**, and does not play (user).
- **A render is never larger than its original** (user). Where the AAC copy would be larger
  (a very short sample, or an original that is already compressed, such as an MP3), the render
  is an exact copy of the original's bytes, stored under the original's own suffix. Of the
  roughly 30,000 renders first made, about 1,900 are such copies, nearly all of them MP3s.
- A render is for auditioning. To use a sound, the user reveals the original in Finder and
  downloads it from the cloud service themselves; the tool never starts a download (user).

Measured size: about 8 GB for 100 hours of audio in 34,000 files.

## Storage (user: "by content, shared copies")

```
/Volumes/DRIVE/audio-embed renders/
  .audio-embed-renders        marker: this folder belongs to the tool
  catalogue.db                copy of the catalogue, refreshed after every render run
  9c/9c41e0…b7.m4a            one render per distinct original, named by its contents
  3f/3fa91c…0e.mp3            a render that is an exact copy keeps the original's suffix

data/renders.db               the catalogue: path, size, mtime -> contents hash, render suffix
data/settings.json            {"renders": "/Volumes/DRIVE/audio-embed renders"}
```

- A render is named by the hash of the original's bytes, the same hash the duplicate finder
  uses. Byte-identical files share one render, and a file that is moved or renamed is matched
  to its existing render the next time it is seen locally, without converting again.
- The catalogue says which render belongs to which indexed file, and which version of the file
  (size and modification time) it was made from.
- The catalogue is what ties an online-only file to its render, so it must not be lost. It is
  copied onto the drive after every run; restoring is copying `catalogue.db` back to
  `data/renders.db`.

## Rules

1. **The audio library stays read-only.** Rendering only reads originals.
2. **The tool writes only inside the render folder, and only if the marker is there.** If the
   drive is not connected, `/Volumes/DRIVE` does not exist; without the marker check a
   render run would quietly create that folder on the Mac's own disk and fill it.
3. **No render is ever thrown away.** The one exception: an AAC render found to be larger than
   its original is removed after an exact copy of the original has taken its place. A render
   whose original was deleted stays as an orphan. Cleaning up is left out of this version on purpose: for an online-only
   file the render cannot be made again without downloading the original.
4. **A file is "local" when it exists and is not empty.** A 0-byte file is a placeholder.
5. **A catalogue row never describes an older version than the index.** If a changed file is
   re-indexed while the drive is away, its row is dropped rather than left pointing at the
   render of the old version.

## How renders get made (user: "I download, it renders")

- `audio-embed renders --to FOLDER` once: creates the folder and its marker, remembers it.
- `audio-embed renders`: renders every indexed file that is local and has no current render.
  Incremental, safe to stop and rerun, several files at once. Ends with a count of the files
  that are online-only and still have no render, by folder.
- `audio-embed index FOLDER` renders the files of that folder at the end of the run when the
  drive is connected, so a folder that was downloaded only to be scanned can be set back to
  online-only straight afterwards.

## Other places that assumed the original is on the Mac

- **Indexing** treated a placeholder as a changed file and printed a failure for it on every
  run. It now counts placeholders as "online-only" and leaves their index entries alone.
- **Waveforms** are looked up by the file's current size and date. For an online-only file the
  stored waveform is used whatever version it was drawn from, and the audio is never read.
- **`audio-embed labels`** rebuilds every label, and reads tags from inside the file. For an
  online-only file the tag-derived labels found earlier are kept.
- **Duplicates** already ignores empty files.

## Page and server

- Each result carries `audio`: `original`, `render` or `none`.
- `/audio/{id}` serves the original, else the render, else answers 404 with a plain sentence.
  A render that is an exact copy of something the browser cannot play (AIFF, CAF, more than
  two channels) is converted on the way out, into the preview cache on the Mac.
- When a local original needs converting (AIFF, CAF, more than two channels) and it has a
  current render, the render is served instead of making a preview.
- `/api/library` reports the render folder and whether it is connected.
- The page shows a small "render" or "online only" marker on a result. How that looks is left
  to the page's visual design.

## Not in this version

- Fetching originals from the cloud service, or releasing them again, from the tool.
- Removing orphaned renders.
- More than one render folder, or a second quality.

## Testing

Unit tests with a temporary folder standing in for the drive: the marker rule, shared renders
for identical files, a new render when a file changes, playing from a render after the original
becomes empty, nothing playable when the folder is away, dropped rows for files changed while
the drive is away, indexing and labels leaving online-only files alone. End to end on a scratch
copy of the index with a scratch render folder, never the real `data/` or the real drive.
