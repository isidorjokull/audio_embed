"""Command line: index a folder, search it by text, find files similar to a file."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from . import audio, generate, renders, search
from .outlines import Outlines, beside
from .store import Store

DEFAULT_DB = Path("data/index.db")
MODEL_NAMES = ("clap", "gemma")
# Optional, next to the index: {"/some/folder": {"source": "foley"}} labels everything under a folder.
LOCATIONS = "locations.json"


def load_embedder(name: str):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import transformers

    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    from . import models

    return models.load(name)


def embed_file(embedder, path: Path):
    """Returns (duration_s, spans in seconds, vectors, waveform outline) for a file.

    Spans and vectors cover the non-silent windows; the outline covers the whole file.
    """
    samples = audio.decode(path, embedder.sr)
    spans = [
        (a, b)
        for a, b in audio.window_spans(len(samples), embedder.sr, embedder.window_s)
        if not audio.is_silent(samples[a:b])
    ]
    batches = [
        embedder.embed_audio([samples[a:b] for a, b in spans[i : i + embedder.batch_size]])
        for i in range(0, len(spans), embedder.batch_size)
    ]
    vecs = np.concatenate(batches) if batches else np.zeros((0, 0), dtype=np.float32)
    seconds = [(a / embedder.sr, b / embedder.sr) for a, b in spans]
    return len(samples) / embedder.sr, seconds, vecs, audio.outline(samples)


def index_path(store: Store) -> Path:
    """The file a store was opened on."""
    return Path(store.db.execute("PRAGMA database_list").fetchone()[2])


def index_folder(store: Store, embedder, root: Path, tag: str | None = None) -> None:
    from tqdm import tqdm

    files = audio.find_audio(root)
    outlines = Outlines(beside(index_path(store)))
    known = store.indexed(embedder.name)
    gone = [p for p in known if Path(p).is_relative_to(root) and not Path(p).exists()]
    store.remove(embedder.name, gone)

    todo = []
    online = 0
    for path in files:
        stat = path.stat()
        if not stat.st_size:
            # A cloud placeholder: nothing to read until it is downloaded. If it was indexed
            # while it was here, that entry stays.
            online += 1
        elif known.get(str(path)) != (stat.st_size, stat.st_mtime):
            todo.append((path, stat))
    print(
        f"[{embedder.name}] {len(files)} audio files, {len(files) - len(todo) - online} already indexed,"
        f" {len(gone)} removed from index, {len(todo)} to embed"
        + (f", {online} online-only" if online else "")
    )

    failed = []
    seconds = 0.0
    for path, stat in tqdm(todo, desc=embedder.name, unit="file", disable=not todo):
        try:
            duration, spans, vecs, levels = embed_file(embedder, path)
        except RuntimeError as e:
            failed.append((path, str(e)))
            continue
        store.put(embedder.name, str(path), stat.st_size, stat.st_mtime, duration, spans, vecs)
        outlines.put(str(path), stat.st_size, stat.st_mtime, levels)
        seconds += duration
    if todo:
        print(f"[{embedder.name}] embedded {seconds / 3600:.2f} h of audio")
    if tag:
        store.tag([str(p) for p in files], tag)
    for path, reason in failed:
        print(f"[{embedder.name}] FAILED {path}: {reason}", file=sys.stderr)


def show(model: str, hits: list[search.Hit], root: str) -> None:
    print(f"\n{model}")
    if not hits:
        print("  (nothing indexed for this model)")
    for n, hit in enumerate(hits, 1):
        minutes, secs = divmod(int(hit.start_s), 60)
        at = f"  @ {minutes}:{secs:02d}" if hit.start_s else ""
        tags = f"  [{', '.join(hit.tags)}]" if hit.tags else ""
        print(f"  {n:2d}. {hit.score:.3f}  {os.path.relpath(hit.path, root)}{at}{tags}")


def common_root(paths: list[str]) -> str:
    return os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])


def render_files(library: renders.Renders, paths: list[str], jobs: int = 6) -> None:
    """Give every local file among `paths` a current render, and say what is still missing.

    Without the drive, nothing can be made; the catalogue rows of files that have changed
    since their render are dropped, so no file is ever played from an older version of itself.
    """
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    have = library.stamps()
    todo, waiting = [], []
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue  # moved or deleted since it was indexed
        if not stat.st_size:
            if path not in have:
                waiting.append(path)
        elif have.get(path) != (stat.st_size, stat.st_mtime):
            todo.append((path, stat.st_size, stat.st_mtime))

    if not library.connected():
        library.forget([path for path, _, _ in todo if path in have])
        print(f"The render folder {library.root} is not there: {len(todo)} files are waiting for a render.")
        return

    def render(item):
        path, size, mtime = item
        try:
            return library.ensure(Path(path), size, mtime)
        except (RuntimeError, OSError):
            return "failed"

    counts = Counter()
    with ThreadPoolExecutor(jobs) as pool:
        for outcome in tqdm(pool.map(render, todo), total=len(todo), desc="renders", unit="file", disable=not todo):
            counts[outcome] += 1
    replaced, left = library.shrink()
    library.back_up()
    print(
        f"renders: {len(have)} in the catalogue, {counts['made']} made,"
        f" {counts['shared']} shared with an identical file"
        + (f", {counts['failed']} could not be read" if counts["failed"] else "")
    )
    if replaced or left:
        print(
            f"{replaced} renders were larger than their originals and are now exact copies of them"
            + (f"; {left} more are waiting for an original to be on this machine" if left else "")
        )
    if waiting:
        print(f"{len(waiting)} files are online-only and have no render yet. Most of them are in:")
        for folder, n in Counter(os.path.dirname(path) for path in waiting).most_common(10):
            print(f"  {n:5d}  {folder}")


def cmd_index(args, store: Store) -> None:
    root = args.folder.resolve()
    if not root.is_dir():
        sys.exit(f"not a folder: {root}")
    for name in args.model or MODEL_NAMES:
        index_folder(store, load_embedder(name), root, args.tag)
    library = renders.beside(args.db)
    if library.root is not None:
        render_files(library, [p for p in store.paths() if Path(p).is_relative_to(root)])


def cmd_renders(args, store: Store) -> None:
    """Make the render of every indexed file that is on this machine and has none yet."""
    settings = args.db.parent / renders.SETTINGS
    if args.to:
        root = args.to.expanduser().resolve()
        try:
            renders.setup(root)
        except RuntimeError as e:
            sys.exit(str(e))
        renders.save_root(settings, root)
    library = renders.beside(args.db)
    if library.root is None:
        sys.exit("No render folder has been chosen. Run `audio-embed renders --to FOLDER` once.")
    if not library.connected():
        sys.exit(f"The render folder {library.root} is not there. Connect the drive and run this again.")
    render_files(library, store.paths(), args.jobs)


def cmd_generator(args, store: Store) -> None:
    """Choose the Stable Audio 3 folder and the folder kept clips go to, and say what is ready."""
    settings = args.db.parent / renders.SETTINGS
    if args.keep_in:
        root = args.keep_in.expanduser().resolve()
        try:
            generate.setup(root)
        except RuntimeError as e:
            sys.exit(str(e))
        generate.save(settings, keep_root=root)
        if generate.label_folder(args.db.parent / LOCATIONS, root):
            print(f'Everything in {root} is now labelled "source: generated" in {LOCATIONS}.')
    if args.sa3:
        generate.save(settings, sa3=args.sa3.expanduser().resolve())
    generator = generate.beside(args.db)
    print(f"Stable Audio 3: {generator.sa3 or 'no folder chosen (--sa3 FOLDER)'}")
    if generator.sa3 is not None:
        for model in generate.MODELS:
            why = generator.problem(model)
            print(f"  {model}: {'ready' if why is None else 'not ready. ' + why}")
    marked = generator.keep_root is not None and (generator.keep_root / generate.MARKER).is_file()
    print(f"Kept clips go to: {generator.keep_root or 'no folder chosen (--keep-in FOLDER)'}"
          + ("" if marked or generator.keep_root is None else "  (its marker is missing, so nothing will be written there)"))


def cmd_search(args, store: Store) -> None:
    for name in args.model or store.models():
        matrix = store.load(name)
        if not matrix.paths:
            show(name, [], "")
            continue
        query = load_embedder(name).embed_text([args.query])[0]
        show(name, search.rank(matrix, query, args.k), common_root(matrix.paths))


def cmd_similar(args, store: Store) -> None:
    path = args.file.resolve()
    if not path.is_file():
        sys.exit(f"not a file: {path}")
    for name in args.model or store.models():
        matrix = store.load(name)
        if not matrix.paths:
            show(name, [], "")
            continue
        query = search.file_vector(matrix, str(path))
        if query is None:
            # Not in the index: embed it now.
            _, _, vecs, _ = embed_file(load_embedder(name), path)
            if not len(vecs):
                sys.exit(f"{path} is silent")
            query = search.unit_mean(vecs)
        hits = search.rank(matrix, query, args.k, exclude=str(path))
        show(name, hits, common_root(matrix.paths))


def cmd_labels(args, store: Store) -> None:
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    from . import labels

    paths = store.paths()
    locations = labels.load_locations(args.db.parent / LOCATIONS)
    earlier = store.labels()
    with ThreadPoolExecutor(8) as pool:
        found = list(tqdm(
            pool.map(lambda p: labels.read(Path(p), locations, earlier.get(p, set())), paths),
            total=len(paths), desc="labels", unit="file",
        ))
    # Written in pieces so an index run going on at the same time is never kept waiting long.
    for i in range(0, len(paths), 1000):
        store.set_labels(dict(zip(paths[i : i + 1000], found[i : i + 1000])))

    files, values = Counter(), {}
    for pairs in found:
        files.update({kind for kind, _ in pairs})
        for kind, value in pairs:
            values.setdefault(kind, Counter())[value] += 1
    print(f"{len(paths)} files")
    for kind, n in files.most_common():
        top = ", ".join(f"{v} ({c})" for v, c in values[kind].most_common(8))
        print(f"  {kind}: {n} files, {len(values[kind])} values: {top}"[:200])


def cmd_outlines(args, store: Store) -> None:
    """Build the waveform outline of every indexed file that does not have a current one."""
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    outlines = Outlines(beside(args.db))
    have = outlines.stamps()
    todo = []
    for path in store.paths():
        try:
            stat = os.stat(path)
        except OSError:
            continue  # moved or deleted since it was indexed
        if stat.st_size and have.get(path) != (stat.st_size, stat.st_mtime):
            todo.append((path, stat.st_size, stat.st_mtime))
    print(f"{len(have)} outlines stored, {len(todo)} to build")

    def build(item):
        try:
            return item, audio.outline(audio.decode(Path(item[0]), 8000))
        except RuntimeError:
            return item, None

    failed = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        for (path, size, mtime), levels in tqdm(pool.map(build, todo), total=len(todo), desc="outlines", unit="file"):
            if levels is None:
                failed += 1
            else:
                outlines.put(path, size, mtime, levels)
    print(f"built {len(todo) - failed}" + (f", {failed} could not be read" if failed else ""))


def cmd_duplicates(args, store: Store) -> None:
    """Write every set of byte-identical indexed files to a spreadsheet, biggest waste first."""
    import csv
    import time

    from . import duplicates

    paths = store.paths()
    # Rendering has hashed most files already; those are not read again.
    found = duplicates.groups(paths, renders.beside(args.db).digests())
    out = args.out or args.db.parent / "duplicates.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # With a byte-order mark, so spreadsheet programs read accented names correctly.
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        rows = csv.writer(f)
        rows.writerow(["set", "copies", "mb_each", "spare_mb", "modified", "folder", "name", "path"])
        for number, (positions, size) in enumerate(found, 1):
            for position in positions:
                path = Path(paths[position])
                modified = time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))
                rows.writerow([
                    number, len(positions), f"{size / 1e6:.2f}", f"{size * (len(positions) - 1) / 1e6:.2f}",
                    modified, str(path.parent), path.name, str(path),
                ])
    spare = sum(size * (len(positions) - 1) for positions, size in found)
    print(
        f"{len(found)} sets of identical files, {sum(len(p) - 1 for p, _ in found)} spare copies,"
        f" {spare / 1e9:.2f} GB that could be freed. Written to {out}"
    )


def cmd_serve(args, store: Store) -> None:
    import threading
    import webbrowser

    import uvicorn

    from .server import create_app

    if not store.models():
        sys.exit("The index is empty. Run `audio-embed index FOLDER` first.")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Audio search is running at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(create_app(args.db), host="127.0.0.1", port=args.port, log_level="warning")


def main() -> None:
    parser = argparse.ArgumentParser(prog="audio-embed", description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="index file (default: %(default)s)")
    sub = parser.add_subparsers(required=True)

    def add(name, fn, help):
        p = sub.add_parser(name, help=help)
        p.set_defaults(fn=fn)
        p.add_argument(
            "--model", action="append", choices=MODEL_NAMES,
            help="model to use; repeat for several (default: all)",
        )
        return p

    p = add("index", cmd_index, "embed every audio file under a folder (only new or changed files)")
    p.add_argument("folder", type=Path)
    p.add_argument("--tag", help="mark every file under the folder with this label, e.g. foley")

    p = add("search", cmd_search, "find files matching a text description")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10, help="number of results (default: %(default)s)")

    p = add("similar", cmd_similar, "find files that sound like a given file")
    p.add_argument("file", type=Path)
    p.add_argument("-k", type=int, default=10, help="number of results (default: %(default)s)")

    p = sub.add_parser(
        "labels", help="read tempo, key, category and more from file names, folders and metadata"
    )
    p.set_defaults(fn=cmd_labels)

    p = sub.add_parser("outlines", help="build the waveform of every indexed file that has none yet")
    p.set_defaults(fn=cmd_outlines)
    p.add_argument("--jobs", type=int, default=6, help="files read at once (default: %(default)s)")

    p = sub.add_parser(
        "renders", help="make a small listening copy of every indexed file, on a drive of your choice"
    )
    p.set_defaults(fn=cmd_renders)
    p.add_argument("--to", type=Path, metavar="FOLDER", help="the folder to keep renders in (asked once, then remembered)")
    p.add_argument("--jobs", type=int, default=6, help="files converted at once (default: %(default)s)")

    p = sub.add_parser("duplicates", help="list every set of byte-identical files in a spreadsheet")
    p.set_defaults(fn=cmd_duplicates)
    p.add_argument("--out", type=Path, metavar="FILE", help="where to write it (default: duplicates.csv next to the index)")

    p = sub.add_parser(
        "generator", help="set up making variations of a file with Stable Audio 3 (asked once, then remembered)"
    )
    p.set_defaults(fn=cmd_generator)
    p.add_argument("--sa3", type=Path, metavar="FOLDER", help="the optimized/mlx folder of a Stable Audio 3 checkout")
    p.add_argument("--keep-in", type=Path, metavar="FOLDER", help="the folder in your library that kept clips are saved to")

    p = sub.add_parser("serve", help="open the search page in your browser")
    p.set_defaults(fn=cmd_serve)
    p.add_argument("--port", type=int, default=8765, help="default: %(default)s")
    p.add_argument("--no-open", action="store_true", help="do not open the browser")

    args = parser.parse_args()
    args.fn(args, Store(args.db))
