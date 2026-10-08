"""Renders: a listening copy of every file, kept on a drive that may be unplugged.

A render is the AAC copy the page already makes for previews. With one, a file
can still be heard after its original has gone online-only. A render is never
larger than its original: where the AAC copy would be (a very short sample, or a
file that is already compressed), the render is an exact copy of the original.

Renders are named by the contents of the original (the hash the duplicate finder
uses), so identical files share one and a moved file finds its render again. A
small catalogue next to the index says which render belongs to which file, and
which version of the file it was made from. For an online-only file the catalogue
is the only thing that ties it to its render, so a copy is kept on the drive.

Two rules keep the drive safe. Nothing is written unless the folder carries the
marker made by `setup`: when the drive is unplugged its mount point is gone, and
writing there anyway would fill the Mac's own disk. And no render is ever thrown
away, because the render of an online-only file cannot be made again; the one
exception is `shrink`, which removes a render only after putting a smaller one
for the same contents in its place.
"""

import json
import os
import shutil
import sqlite3
import threading
from pathlib import Path

from . import audio, duplicates

MARKER = ".audio-embed-renders"
# Both live next to the index.
SETTINGS = "settings.json"
CATALOGUE = "renders.db"
# The catalogue's copy inside the render folder.
BACKUP = "catalogue.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS renders (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    digest TEXT NOT NULL,
    suffix TEXT NOT NULL DEFAULT '.m4a'
)
"""
# The suffix of the AAC copy. A render that is an exact copy keeps its original's suffix.
AAC = ".m4a"


def setup(root: Path) -> None:
    """Make `root` the render folder. The folder it sits in (the drive) must be there."""
    if not root.parent.is_dir():
        raise RuntimeError(f"{root.parent} is not there. Is the drive connected?")
    root.mkdir(exist_ok=True)
    (root / MARKER).touch()


def saved_root(settings: Path) -> Path | None:
    """The render folder chosen earlier, if any."""
    if not settings.exists():
        return None
    folder = json.loads(settings.read_text(encoding="utf-8")).get("renders")
    return Path(folder) if folder else None


def save_root(settings: Path, root: Path) -> None:
    known = json.loads(settings.read_text(encoding="utf-8")) if settings.exists() else {}
    known["renders"] = str(root)
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(known, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unchanged(path: str, size: int, mtime: float) -> bool:
    try:
        stat = os.stat(path)
    except OSError:
        return False
    return (stat.st_size, stat.st_mtime) == (size, mtime)


def beside(db_path: Path) -> "Renders":
    """The renders that go with an index."""
    return Renders(db_path.parent / CATALOGUE, saved_root(db_path.parent / SETTINGS))


class Renders:
    def __init__(self, catalogue: Path, root: Path | None, transcode=audio.write_preview) -> None:
        catalogue.parent.mkdir(parents=True, exist_ok=True)
        self.catalogue = catalogue
        self.root = root
        self.transcode = transcode
        # One connection shared by the server's worker threads, guarded by a lock.
        self.db = sqlite3.connect(catalogue, timeout=30, check_same_thread=False)
        self.db.execute(SCHEMA)
        if "suffix" not in [column[1] for column in self.db.execute("PRAGMA table_info(renders)")]:
            self.db.execute(f"ALTER TABLE renders ADD COLUMN suffix TEXT NOT NULL DEFAULT '{AAC}'")
        self._lock = threading.Lock()

    def connected(self) -> bool:
        """Whether the render folder is within reach right now."""
        return self.root is not None and (self.root / MARKER).is_file()

    def file(self, digest: str, suffix: str = AAC) -> Path:
        return self.root / digest[:2] / f"{digest}{suffix}"

    def _stored(self, digest: str) -> str | None:
        """The suffix of the render these contents already have on the drive, if any."""
        with self._lock:
            suffixes = [s for (s,) in self.db.execute("SELECT DISTINCT suffix FROM renders WHERE digest = ?", (digest,))]
        return next((s for s in [*suffixes, AAC] if self.file(digest, s).is_file()), None)

    def _write(self, source: Path, size: int, digest: str) -> str:
        """Put the render of `source` on the drive and return its suffix.

        The AAC copy, unless that comes out larger than the original: then an exact copy.
        """
        folder = self.root / digest[:2]
        folder.mkdir(exist_ok=True)
        # Two identical files may be converted at once; each writes its own partial file.
        partial = folder / f"{digest}.{threading.get_ident()}.partial"
        try:
            self.transcode(source, partial)
            suffix = AAC
            if partial.stat().st_size > size:
                shutil.copyfile(source, partial)
                suffix = source.suffix.lower()
            os.replace(partial, self.file(digest, suffix))
        finally:
            partial.unlink(missing_ok=True)
        return suffix

    def stamps(self) -> dict[str, tuple[int, float]]:
        """path -> (size, mtime) of the file version each render was made from."""
        with self._lock:
            return {path: (size, mtime) for path, size, mtime in self.db.execute("SELECT path, size, mtime FROM renders")}

    def digests(self) -> dict[tuple[str, int, float], str]:
        """(path, size, mtime) -> contents hash, in the shape `duplicates.groups` takes as known hashes."""
        with self._lock:
            rows = self.db.execute("SELECT path, size, mtime, digest FROM renders")
            return {(path, size, mtime): digest for path, size, mtime, digest in rows}

    def copy_of(self, path: str, stamp: tuple[int, float] | None = None) -> Path | None:
        """The render of a file, if it has one and the drive is connected.

        With `stamp` (size, mtime), only a render made from that version of the file counts.
        """
        if not self.connected():
            return None
        with self._lock:
            row = self.db.execute("SELECT size, mtime, digest, suffix FROM renders WHERE path = ?", (path,)).fetchone()
        if row is None or (stamp is not None and tuple(row[:2]) != tuple(stamp)):
            return None
        out = self.file(row[2], row[3])
        return out if out.is_file() else None

    def ensure(self, path: Path, size: int, mtime: float) -> str:
        """Give a local file its render. Returns "made", or "shared" if identical contents already had one."""
        if not self.connected():
            raise RuntimeError(f"The render folder {self.root} is not there. Is the drive connected?")
        digest = duplicates.digest(str(path))
        suffix, outcome = self._stored(digest), "shared"
        if suffix is None:
            suffix, outcome = self._write(path, size, digest), "made"
        with self._lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO renders VALUES (?, ?, ?, ?, ?)", (str(path), size, mtime, digest, suffix)
            )
        return outcome

    def shrink(self) -> tuple[int, int]:
        """Replace every AAC render that is larger than its original with an exact copy of the original.

        Returns (replaced, left): a render is left as it is while no original with
        those contents is on this machine to copy.
        """
        if not self.connected():
            return 0, 0
        with self._lock:
            rows = self.db.execute("SELECT digest, path, size, mtime FROM renders WHERE suffix = ?", (AAC,)).fetchall()
        originals: dict[str, list[tuple[str, int, float]]] = {}
        for digest, path, size, mtime in rows:
            originals.setdefault(digest, []).append((path, size, mtime))
        replaced = left = 0
        for digest, files in originals.items():
            old = self.file(digest)
            if not old.is_file() or old.stat().st_size <= files[0][1]:
                continue
            source = next((Path(p) for p, size, mtime in files if _unchanged(p, size, mtime)), None)
            if source is None:
                left += 1
                continue
            new = self.file(digest, source.suffix.lower())
            partial = old.with_name(f"{digest}.{threading.get_ident()}.partial")
            try:
                shutil.copyfile(source, partial)
                os.replace(partial, new)
            finally:
                partial.unlink(missing_ok=True)
            with self._lock, self.db:
                self.db.execute("UPDATE renders SET suffix = ? WHERE digest = ?", (new.suffix, digest))
            if new != old:  # an original that is itself .m4a has just taken the old render's place
                old.unlink()
            replaced += 1
        return replaced, left

    def forget(self, paths: list[str]) -> None:
        """Drop catalogue rows. The renders themselves stay on the drive."""
        with self._lock, self.db:
            self.db.executemany("DELETE FROM renders WHERE path = ?", [(path,) for path in paths])

    def back_up(self) -> None:
        """Copy the catalogue into the render folder, so losing data/ does not orphan the renders."""
        if not self.connected():
            return
        partial = self.root / f"{BACKUP}.partial"
        partial.unlink(missing_ok=True)
        copy = sqlite3.connect(partial)
        with self._lock:
            self.db.backup(copy)
        copy.close()
        os.replace(partial, self.root / BACKUP)
