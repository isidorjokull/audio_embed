"""Thumbs up and down on search results, kept as training data for later.

Stored apart from the index, one JSON object per line and keyed by file path,
so the votes survive deleting or rebuilding the index. The file is append-only:
changing or clearing a vote adds a line, and the latest line for a
(query, file) pair is the one that counts.
"""

import json
import threading
import time
from pathlib import Path


def query_key(query: dict) -> str:
    """Identifies a search: {"kind": "text", "text": ...} or {"kind": "like", "path": ...}."""
    return f"text:{query['text']}" if query["kind"] == "text" else f"like:{query['path']}"


class Feedback:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._votes: dict[tuple[str, str], int] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    self._apply(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    continue  # a line cut short by a crash; the rest of the file is still good

    def _apply(self, record: dict) -> None:
        key = (query_key(record["query"]), record["path"])
        if record["verdict"]:
            self._votes[key] = record["verdict"]
        else:
            self._votes.pop(key, None)

    def record(self, query: dict, path: str, verdict: int, **context) -> None:
        """Save a vote: 1 for a good match, -1 for a bad one, 0 to clear it.

        `context` is whatever describes how the result was shown (model, rank,
        score, matched window), kept so the votes can also compare models.
        """
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "query": query,
            "path": path,
            "verdict": verdict,
            **context,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._apply(record)

    def verdict(self, query: dict, path: str) -> int:
        return self._votes.get((query_key(query), path), 0)

    def votes(self, query: dict) -> dict[str, int]:
        """Every standing vote for one search: file path -> 1 or -1."""
        key = query_key(query)
        return {path: verdict for (asked, path), verdict in self._votes.items() if asked == key}

    def by_text(self) -> dict[str, dict[str, int]]:
        """Every text search with a standing vote: its text -> {file path: 1 or -1}."""
        found: dict[str, dict[str, int]] = {}
        with self._lock:
            for (asked, path), verdict in self._votes.items():
                if asked.startswith("text:"):
                    found.setdefault(asked.removeprefix("text:"), {})[path] = verdict
        return found
