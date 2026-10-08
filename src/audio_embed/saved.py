"""Things saved from the page: collections of files, and mood presets.

Both live next to the index (in data/) and neither touches the audio library.
"""

import json
import os
from pathlib import Path

DEFAULT_MOODS = [
    {"name": "dark", "description": "a dark, ominous low drone"},
    {"name": "tense", "description": "tense, unsettling, eerie atmosphere"},
    {"name": "warm", "description": "a warm, soft, gentle pad"},
    {"name": "bright", "description": "bright, shimmering, glassy tones"},
    {"name": "calm", "description": "calm, peaceful, quiet ambience"},
    {"name": "harsh", "description": "harsh, distorted, aggressive noise"},
]


def clean_name(name: str) -> str | None:
    """A name usable as a folder name, or None if it is not."""
    name = " ".join(str(name).split())
    if not name or len(name) > 60 or name.startswith(".") or any(c in name for c in "/:\\\0"):
        return None
    return name


class Collections:
    """Each collection is a folder of shortcuts (symlinks) to the original files.

    The folder is the collection: it can be opened in Finder or added to a DAW's
    browser, and deleting a shortcut there takes the file out of the collection.
    The audio is never copied, moved or changed, and only shortcuts are ever removed.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def folder(self, name: str) -> Path:
        return self.root / name

    def names(self) -> list[tuple[str, int]]:
        """Every collection with the number of files in it."""
        if not self.root.is_dir():
            return []
        return sorted(
            (d.name, len(self._links(d.name))) for d in self.root.iterdir() if d.is_dir() and clean_name(d.name)
        )

    def _links(self, name: str) -> list[Path]:
        folder = self.folder(name)
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.is_symlink())

    def paths(self, name: str) -> list[str]:
        """The original files a collection points at."""
        return [os.readlink(link) for link in self._links(name)]

    def add(self, name: str, path: Path) -> None:
        if str(path) in self.paths(name):
            return
        folder = self.folder(name)
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / path.name
        number = 2
        while link.exists() or link.is_symlink():
            link = folder / f"{path.stem} ({number}){path.suffix}"
            number += 1
        link.symlink_to(path)

    def remove(self, name: str, path: Path) -> None:
        for link in self._links(name):
            if os.readlink(link) == str(path):
                link.unlink()


class Moods:
    """Mood words mapped to a description that finds them, e.g. tense -> "tense, unsettling, eerie atmosphere"."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def all(self) -> list[dict]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return [dict(m) for m in DEFAULT_MOODS]

    def _write(self, moods: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(moods, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def save(self, name: str, description: str) -> None:
        """Add a mood, or change the description of an existing one."""
        moods = [m for m in self.all() if m["name"] != name]
        self._write(moods + [{"name": name, "description": description}])

    def remove(self, name: str) -> None:
        self._write([m for m in self.all() if m["name"] != name])
