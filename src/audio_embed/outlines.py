"""Waveform outlines: the shape the page draws for each file.

Stored in their own SQLite file next to the index, one row per file path: up to
800 peak levels as single bytes, scaled so the loudest point is 255. They are
kept out of index.db on purpose, so that saving an outline while someone is
browsing never makes the server think the index changed and re-read it.

It is a cache keyed by path, size and modification time. Delete the file and
`audio-embed outlines`, indexing, or simply browsing rebuilds it.
"""

import sqlite3
import threading
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS outlines (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    levels BLOB NOT NULL
)
"""


def beside(db_path: Path) -> Path:
    """Where the outlines for an index live."""
    return db_path.parent / "outlines.db"


class Outlines:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the server's worker threads, guarded by a lock.
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.execute(SCHEMA)
        self._lock = threading.Lock()

    def put(self, path: str, size: int, mtime: float, levels: np.ndarray) -> None:
        """`levels` is what `audio.outline` returns: one byte per bucket."""
        with self._lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO outlines VALUES (?, ?, ?, ?)",
                (path, size, mtime, levels.astype(np.uint8).tobytes()),
            )

    def get(self, path: str, size: int, mtime: float) -> list[float] | None:
        """Levels from 0 to 1, or None if there is no outline for this version of the file."""
        with self._lock:
            row = self.db.execute(
                "SELECT levels FROM outlines WHERE path = ? AND size = ? AND mtime = ?", (path, size, mtime)
            ).fetchone()
        if row is None:
            return None
        return (np.frombuffer(row[0], dtype=np.uint8) / 255).round(3).tolist()

    def latest(self, path: str) -> list[float] | None:
        """The stored outline of a path whatever version it was drawn from.

        For a file that has gone online-only: its size on disk is now 0, so `get` would miss.
        """
        with self._lock:
            row = self.db.execute("SELECT levels FROM outlines WHERE path = ?", (path,)).fetchone()
        if row is None:
            return None
        return (np.frombuffer(row[0], dtype=np.uint8) / 255).round(3).tolist()

    def stamps(self) -> dict[str, tuple[int, float]]:
        """path -> (size, mtime) of the file version each stored outline was made from."""
        with self._lock:
            return {path: (size, mtime) for path, size, mtime in self.db.execute("SELECT path, size, mtime FROM outlines")}
