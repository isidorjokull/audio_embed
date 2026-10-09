"""Files left out of every search because the user hid them.

data/hidden.json holds what was hidden by hand: single files, and folders, which
hide everything under them (files indexed later too). It is kept apart from the
index and keyed by path, so it survives re-indexing and `audio-embed labels`.
Nothing here touches the audio.

Some files are given away as finished songs by their names ("Artist - Title.mp3").
Those are only suggested for hiding, never hidden on their own, because the rule
is wrong about one file in ten.
"""

import json
import os
import re
import threading
from pathlib import Path

import numpy as np

from .labels import nfc

# What marks a name as a song. Chosen on the user's own library: of the 92 files it picks
# out there, about 82 are songs. Embedded tags were no help (sound-effect libraries fill in
# artist, album and genre as well), and CLAP hears the user's own cello and piano bounces
# as music just the same, so neither is used.
SONG_SHORTEST_S = 60
# Words left in a name by wherever the song was downloaded from. "M2W" ends a file converted from MP3.
SONG_WORDS = re.compile(r"official (audio|video|music|lyric)|\bfeat\b|\bft\.|soundtrack|karaoke|M2W$|\(live\)|radio edit", re.I)
# "01 - Artist - Title" and "Artist - Title". The user's own bounces are named this way too,
# so these count only for a file in a format music is passed around in.
SONG_TRACK = re.compile(r"^\d{1,2}[ ._-]+\S.* - \S")
SONG_PAIR = re.compile(r"^[^\W\d_][^_]*\S - [^\W\d_][^_]*$")
SONG_FORMATS = {".mp3", ".flac", ".m4a", ".ogg", ".opus"}
# Folder names that say what is in them, within this many folders above the file.
SONG_FOLDERS = {"lög", "song", "songs", "önnur tónlist", "mp3 to waw"}
SONG_FOLDERS_ABOVE = 3


def named_like_song(path: str) -> bool:
    file = Path(nfc(path))
    if SONG_WORDS.search(file.stem):
        return True
    if file.suffix.lower() in SONG_FORMATS and (SONG_TRACK.search(file.stem) or SONG_PAIR.search(file.stem)):
        return True
    return any(part.casefold() in SONG_FOLDERS for part in file.parent.parts[-SONG_FOLDERS_ABOVE:])


def looks_like_song(path: str, duration_s: float) -> bool:
    """Whether a file's name and length say it is a finished song rather than a sound to work with."""
    return duration_s >= SONG_SHORTEST_S and named_like_song(path)


class Hidden:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        # Goes up with every change, so what was worked out from an earlier state can be told apart.
        self.version = 0
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._files = {nfc(p) for p in saved.get("files", [])}
        self._folders = {nfc(p).rstrip("/") for p in saved.get("folders", [])}
        # Files that look like songs by name and were said not to be (or to be wanted anyway).
        self._not_songs = {nfc(p) for p in saved.get("not_songs", [])}

    def _write(self) -> None:
        self.version += 1
        body = {"files": sorted(self._files), "folders": sorted(self._folders), "not_songs": sorted(self._not_songs)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the file and moved into place: this list is made by hand and cannot be rebuilt.
        partial = self.path.with_suffix(".partial")
        partial.write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        os.replace(partial, self.path)

    def files(self) -> list[str]:
        return sorted(self._files)

    def folders(self) -> list[str]:
        return sorted(self._folders)

    def hide(self, path: str) -> None:
        with self._lock:
            self._files.add(nfc(path))
            self._write()

    def hide_all(self, paths: list[str]) -> None:
        with self._lock:
            self._files |= {nfc(p) for p in paths}
            self._write()

    def show(self, path: str) -> None:
        """Stop hiding one file. If its name looks like a song, it is not suggested for hiding again."""
        with self._lock:
            self._files.discard(nfc(path))
            if named_like_song(path):
                self._not_songs.add(nfc(path))
            self._write()

    def hide_folder(self, folder: str) -> None:
        with self._lock:
            self._folders.add(nfc(folder).rstrip("/"))
            self._write()

    def show_folder(self, folder: str) -> None:
        with self._lock:
            self._folders.discard(nfc(folder).rstrip("/"))
            self._write()

    def not_a_song(self, path: str) -> None:
        with self._lock:
            self._not_songs.add(nfc(path))
            self._write()

    def folder_of(self, path: str) -> str | None:
        """The hidden folder a file is under, if it is under one."""
        path = nfc(path)
        return next((folder for folder in sorted(self._folders) if path.startswith(folder + "/")), None)

    def mask(self, paths: list[str]) -> np.ndarray:
        """True for each path that is hidden, on its own or by its folder."""
        folders = tuple(folder + "/" for folder in self._folders)
        return np.array([p in self._files or p.startswith(folders) for p in map(nfc, paths)], dtype=bool)

    def suggested(self, paths: list[str], durations: list[float]) -> list[int]:
        """Positions of the files that look like songs and have been neither hidden nor cleared."""
        hidden = self.mask(paths)
        return [
            i for i, (path, duration) in enumerate(zip(paths, durations))
            if not hidden[i] and looks_like_song(path, duration) and nfc(path) not in self._not_songs
        ]
