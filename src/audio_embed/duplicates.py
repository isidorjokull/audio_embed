"""Finding exact copies: files whose contents are byte-for-byte identical.

Embeddings are deliberately not used. They cannot tell two adjacent notes of
the same bass patch apart, so "sounds almost identical" produced false matches
on real sample packs. Comparing contents is never wrong: files are grouped by
size, and only files that share a size with another are read and hashed.
"""

import hashlib
import os

Stamp = tuple[str, int, float]  # path, size, modification time


def digest(path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def groups(paths: list[str], known: dict[Stamp, str] | None = None) -> list[tuple[list[int], int]]:
    """Sets of identical files as (positions in `paths`, size of each file in bytes).

    Largest saving first. `known` remembers hashes between calls, keyed by path,
    size and modification time, so unchanged files are not read twice.
    """
    known = {} if known is None else known
    by_size: dict[int, list[tuple[int, Stamp]]] = {}
    for position, path in enumerate(paths):
        try:
            stat = os.stat(path)
        except OSError:
            continue  # moved or deleted since it was indexed
        if stat.st_size:  # empty files are placeholders, not copies of each other
            by_size.setdefault(stat.st_size, []).append((position, (path, stat.st_size, stat.st_mtime)))

    found = []
    for size, same_size in by_size.items():
        if len(same_size) < 2:
            continue
        by_content: dict[str, list[int]] = {}
        for position, stamp in same_size:
            if stamp not in known:
                try:
                    known[stamp] = digest(stamp[0])
                except OSError:
                    continue
            by_content.setdefault(known[stamp], []).append(position)
        found += [(positions, size) for positions in by_content.values() if len(positions) > 1]
    return sorted(found, key=lambda group: (-group[1] * (len(group[0]) - 1), group[0][0]))
