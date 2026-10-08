"""The index: one SQLite file holding file records and one vector per window.

Vectors are stored as float16 blobs (half the size, no measurable effect on
ranking) and loaded into a single float32 matrix for search.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS indexed (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    duration_s REAL NOT NULL,
    PRIMARY KEY (file_id, model)
);
CREATE TABLE IF NOT EXISTS windows (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS windows_by_model ON windows (model, file_id);
CREATE TABLE IF NOT EXISTS tags (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (file_id, tag)
);
CREATE TABLE IF NOT EXISTS labels (
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (file_id, kind, value)
);
"""


@dataclass
class Matrix:
    """Every window vector for one model, with the file each row belongs to."""

    vecs: np.ndarray  # (n_windows, dim) float32, unit length
    file_idx: np.ndarray  # (n_windows,) index into `paths`
    starts: np.ndarray  # (n_windows,) seconds
    paths: list[str]
    ids: list[int]  # file id for each entry of `paths`
    durations: list[float]  # seconds, for each entry of `paths`
    tags: list[list[str]]  # labels such as "foley", for each entry of `paths`
    labels: list[dict[str, list[str]]]  # kind -> values, e.g. {"bpm": ["82"]}, for each entry of `paths`


class Store:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    def indexed(self, model: str) -> dict[str, tuple[int, float]]:
        """path -> (size, mtime) as they were when the file was last indexed."""
        rows = self.db.execute(
            "SELECT f.path, i.size, i.mtime FROM indexed i JOIN files f ON f.id = i.file_id"
            " WHERE i.model = ?",
            (model,),
        )
        return {path: (size, mtime) for path, size, mtime in rows}

    def put(self, model, path, size, mtime, duration_s, spans, vecs) -> None:
        """Replace everything stored for (path, model). `spans` are (start_s, end_s)."""
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO files (path) VALUES (?)", (path,))
            (file_id,) = self.db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
            self.db.execute(
                "DELETE FROM windows WHERE file_id = ? AND model = ?", (file_id, model)
            )
            self.db.execute(
                "INSERT OR REPLACE INTO indexed VALUES (?, ?, ?, ?, ?)",
                (file_id, model, size, mtime, duration_s),
            )
            self.db.executemany(
                "INSERT INTO windows VALUES (?, ?, ?, ?, ?)",
                [
                    (file_id, model, start, end, vec.astype(np.float16).tobytes())
                    for (start, end), vec in zip(spans, vecs)
                ],
            )

    def remove(self, model: str, paths: list[str]) -> None:
        with self.db:
            for path in paths:
                row = self.db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
                if row:
                    self.db.execute(
                        "DELETE FROM windows WHERE file_id = ? AND model = ?", (row[0], model)
                    )
                    self.db.execute(
                        "DELETE FROM indexed WHERE file_id = ? AND model = ?", (row[0], model)
                    )

    def tag(self, paths: list[str], tag: str) -> None:
        """Mark files with a label. Tags belong to the file, not to a model."""
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO tags SELECT id, ? FROM files WHERE path = ?",
                [(tag, path) for path in paths],
            )

    def paths(self) -> list[str]:
        """Every file that is indexed under at least one model."""
        rows = self.db.execute(
            "SELECT path FROM files WHERE id IN (SELECT file_id FROM indexed) ORDER BY path"
        )
        return [path for (path,) in rows]

    def labels(self) -> dict[str, set[tuple[str, str]]]:
        """path -> its (kind, value) labels, for every file that has any."""
        out: dict[str, set[tuple[str, str]]] = {}
        rows = self.db.execute("SELECT f.path, l.kind, l.value FROM labels l JOIN files f ON f.id = l.file_id")
        for path, kind, value in rows:
            out.setdefault(path, set()).add((kind, value))
        return out

    def set_labels(self, labels: dict[str, set[tuple[str, str]]]) -> None:
        """Replace the (kind, value) labels of each path. Paths not in the index are ignored."""
        with self.db:
            for path, pairs in labels.items():
                row = self.db.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
                if row:
                    self.db.execute("DELETE FROM labels WHERE file_id = ?", row)
                    self.db.executemany(
                        "INSERT INTO labels VALUES (?, ?, ?)",
                        [(row[0], kind, value) for kind, value in pairs],
                    )

    def models(self) -> list[str]:
        return [m for (m,) in self.db.execute("SELECT DISTINCT model FROM indexed ORDER BY model")]

    def load(self, model: str) -> Matrix:
        rows = self.db.execute(
            "SELECT f.id, f.path, i.duration_s, w.start_s, w.vec FROM windows w"
            " JOIN files f ON f.id = w.file_id"
            " JOIN indexed i ON i.file_id = w.file_id AND i.model = w.model"
            " WHERE w.model = ? ORDER BY f.path, w.start_s",
            (model,),
        ).fetchall()
        paths: list[str] = []
        ids: list[int] = []
        durations: list[float] = []
        file_idx = np.empty(len(rows), dtype=np.int64)
        for i, (file_id, path, duration, _, _) in enumerate(rows):
            if not paths or paths[-1] != path:
                paths.append(path)
                ids.append(file_id)
                durations.append(duration)
            file_idx[i] = len(paths) - 1
        vecs = (
            np.stack([np.frombuffer(r[4], dtype=np.float16) for r in rows]).astype(np.float32)
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        starts = np.array([r[3] for r in rows], dtype=np.float64)
        tagged: dict[int, list[str]] = {}
        for file_id, tag in self.db.execute("SELECT file_id, tag FROM tags ORDER BY tag"):
            tagged.setdefault(file_id, []).append(tag)
        tags = [tagged.get(file_id, []) for file_id in ids]
        labelled: dict[int, dict[str, list[str]]] = {}
        for file_id, kind, value in self.db.execute(
            "SELECT file_id, kind, value FROM labels ORDER BY kind, value"
        ):
            labelled.setdefault(file_id, {}).setdefault(kind, []).append(value)
        labels = [labelled.get(file_id, {}) for file_id in ids]
        return Matrix(vecs, file_idx, starts, paths, ids, durations, tags, labels)
