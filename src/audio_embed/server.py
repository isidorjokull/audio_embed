"""The local browser interface: a small server that keeps the models loaded.

Bound to 127.0.0.1 only. Audio is served by file id, so only indexed files are
reachable, and the library itself is never written to. A file whose original has
gone online-only is played from its render when the render drive is connected.
"""

import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from . import audio, classify, duplicates, facets, models, renders, search
from .cli import common_root, load_embedder
from .feedback import Feedback
from .outlines import Outlines, beside
from .saved import Collections, Moods, clean_name
from .store import Matrix, Store

PAGE = Path(__file__).parent / "web" / "index.html"
# Files a browser cannot play are converted on first play; the oldest copies are dropped past this size.
PREVIEW_CACHE_BYTES = 2 * 1024**3
# Formats served straight from disk (with up to two channels); anything else is converted.
# How often the index is re-read at most while something is writing to it.
RELOAD_EVERY_S = 15
# How hard the user's votes pull a search, relative to the search itself. On the first
# 89 real votes, 2 ranked a held-out liked file above a held-out rejected one in 97% of
# pairs, against 82% with no steering; 1 and 4 were slightly worse.
STEER_WEIGHT = 2.0
# The duplicate finder shows the sets that would free the most space.
DUPLICATE_SETS_SHOWN = 200
# How many stored waveform outlines one request may ask for.
OUTLINES_PER_REQUEST = 60
BROWSER_PLAYABLE = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
}


class Library:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.cache = db_path.parent / "previews"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.outlines = Outlines(beside(db_path))
        self.renders = renders.beside(db_path)
        self._matrices: dict[str, Matrix] = {}
        self._loaded_mtime = None
        self._loaded_at = 0.0
        self.categories = classify.load_categories(db_path.parent / "categories.json")
        self._category_vectors = None
        self._facets = {}  # id(matrix) -> (matrix, sound category per file, filter index)
        self._vectors = {}  # id(matrix) -> (matrix, one vector per file, path -> position)
        self._roots = {}  # id(matrix) -> (matrix, the folder all its files share)
        self._duplicates = {}  # id(matrix) -> (matrix, sets of identical files)
        self._digests = {}  # content hashes, kept across reloads of the index
        self._embedders = {}
        self._text_vectors = {}
        self._lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._preview_lock = threading.Lock()

    def matrices(self) -> dict[str, Matrix]:
        """Every model's vectors, reloaded when the index file changes.

        While an indexing run is writing, the file changes constantly, so reloads
        are spaced out; new files show up within RELOAD_EVERY_S seconds.
        """
        with self._lock:
            mtime = self.db_path.stat().st_mtime
            waited = time.monotonic() - self._loaded_at > RELOAD_EVERY_S
            if mtime != self._loaded_mtime and (waited or not self._matrices):
                store = Store(self.db_path)
                self._matrices = {name: store.load(name) for name in store.models()}
                self._matrices = {n: m for n, m in self._matrices.items() if m.paths}
                self._loaded_mtime = mtime
                self._loaded_at = time.monotonic()
                self._facets, self._vectors, self._duplicates, self._roots = {}, {}, {}, {}
            return self._matrices

    def widest(self) -> tuple[str, Matrix] | None:
        """The model that covers the most files, used for views that are not a search."""
        matrices = self.matrices()
        if not matrices:
            return None
        name = max(matrices, key=lambda n: len(matrices[n].paths))
        return name, matrices[name]

    def root(self, matrix: Matrix) -> str:
        """The folder every file of a matrix is under."""
        cached = self._roots.get(id(matrix))
        if cached is None or cached[0] is not matrix:
            cached = (matrix, common_root(matrix.paths))
            self._roots[id(matrix)] = cached
        return cached[1]

    def vectors(self, matrix: Matrix):
        """(one vector per file, path -> position in the matrix)."""
        cached = self._vectors.get(id(matrix))
        if cached is None or cached[0] is not matrix:
            cached = (matrix, search.file_means(matrix), {p: i for i, p in enumerate(matrix.paths)})
            self._vectors[id(matrix)] = cached
        return cached[1], cached[2]

    def duplicates(self, matrix: Matrix) -> list[tuple[list[int], int]]:
        cached = self._duplicates.get(id(matrix))
        if cached is None or cached[0] is not matrix:
            cached = (matrix, duplicates.groups(matrix.paths, self._digests))
            self._duplicates[id(matrix)] = cached
        return cached[1]

    def ready(self, name: str) -> bool:
        return name in self._embedders

    def filters(self, matrix: Matrix):
        """(sound category per file, filter index) for a matrix.

        The sound category comes from CLAP (see classify.py), so it is None for
        files CLAP has not indexed. Every other kind comes from `matrix.labels`.
        """
        cached = self._facets.get(id(matrix))
        if cached is not None and cached[0] is matrix:
            return cached[1], cached[2]
        clap = self.matrices().get(classify.MODEL)
        sound_of = {}
        if clap is not None:
            if self._category_vectors is None:
                embedder = self.embedder(classify.MODEL)
                with self._model_lock:
                    self._category_vectors = embedder.embed_text([d for _, d in self.categories])
            best = classify.assign(search.file_means(clap), self._category_vectors)
            sound_of = {path: self.categories[b][0] for path, b in zip(clap.paths, best)}
        sounds = [sound_of.get(path) for path in matrix.paths]
        extra = getattr(matrix, "labels", None) or [{}] * len(matrix.paths)
        per_file = [
            {**labels, **({"sound": [sound]} if sound else {})} for sound, labels in zip(sounds, extra)
        ]
        for labels in per_file:
            if "bpm" in labels:
                labels["bpm"] = sorted({r for r in map(facets.tempo_range, labels["bpm"]) if r})
        index = facets.build(per_file)
        self._facets[id(matrix)] = (matrix, sounds, index)
        return sounds, index

    def embedder(self, name: str):
        with self._model_lock:
            if name not in self._embedders:
                self._embedders[name] = load_embedder(name)
            return self._embedders[name]

    def text_vector(self, model: str, text: str):
        key = (model, text)
        if key not in self._text_vectors:
            embedder = self.embedder(model)
            with self._model_lock:
                self._text_vectors[key] = embedder.embed_text([text])[0]
        return self._text_vectors[key]

    def path_of(self, file_id: int) -> Path | None:
        for matrix in self.matrices().values():
            if file_id in matrix.ids:
                return Path(matrix.paths[matrix.ids.index(file_id)])
        return None

    def heard_from(self, path: Path) -> tuple[str, Path | None]:
        """Where a file can be listened to: ("original", path), ("render", its copy) or ("none", None)."""
        if audio.is_local(path):
            return "original", path
        copy = self.renders.copy_of(str(path))
        return ("render", copy) if copy else ("none", None)

    def cached(self, file_id: int, path: Path, suffix: str) -> Path:
        # Recreated on demand, so deleting the folder to free space is safe while running.
        self.cache.mkdir(parents=True, exist_ok=True)
        stat = path.stat()
        return self.cache / f"{file_id}-{stat.st_size}-{int(stat.st_mtime)}{suffix}"

    def preview(self, file_id: int, path: Path) -> Path:
        out = self.cached(file_id, path, ".m4a")
        with self._preview_lock:
            if out.exists():
                os.utime(out)
                return out
            partial = out.with_suffix(".partial")
            audio.write_preview(path, partial)
            partial.rename(out)
            self._trim_previews()
        return out

    def _trim_previews(self) -> None:
        files = sorted(self.cache.glob("*.m4a"), key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for f in files:
            total += f.stat().st_size
            if total > PREVIEW_CACHE_BYTES:
                f.unlink()

    def peaks(self, path: Path, build: bool = True) -> list[float] | None:
        """A file's waveform outline. Normally stored already; built and stored here if not.

        With `build` off, a missing outline gives None instead of reading the audio.
        An online-only file has no audio to read: it gets the outline stored while it was here.
        """
        if not audio.is_local(path):
            return self.outlines.latest(str(path))
        stat = path.stat()
        levels = self.outlines.get(str(path), stat.st_size, stat.st_mtime)
        if levels is None and build:
            made = audio.outline(audio.decode(path, 8000))
            self.outlines.put(str(path), stat.st_size, stat.st_mtime, made)
            levels = (made / 255).round(3).tolist()
        return levels


def hits_json(
    matrix: Matrix, model: str, hits: list[search.Hit], vote=lambda path: 0, sounds=None, root=None,
    heard=lambda path: "original",
) -> list[dict]:
    """`vote` gives the saved thumbs up (1) or down (-1) for a path under the current search.

    `sounds` is the sound category of each file of the matrix, when known. `root` is the
    folder all of the matrix's files share; pass it when known, since finding it reads every path.
    `heard` says what a path plays from: "original", "render", or "none" when it cannot be played.
    """
    root = root or common_root(matrix.paths)
    out = []
    for hit in hits:
        folder = os.path.relpath(os.path.dirname(hit.path), root)
        out.append({
            "id": matrix.ids[hit.index],
            "name": Path(hit.path).stem,
            "folder": "" if folder == "." else folder,
            # The two nearest folder names: enough to place a file when the library root is far up.
            "near": "/".join(Path(hit.path).parent.parts[-2:]),
            "score": round(hit.score, 3),
            "start": hit.start_s,
            "duration": matrix.durations[hit.index],
            "window": models.EMBEDDERS[model].window_s,
            "tags": hit.tags,
            "sound": sounds[hit.index] if sounds else None,
            "vote": vote(hit.path),
            "audio": heard(hit.path),
        })
    return out


def error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(db_path: Path) -> Starlette:
    library = Library(db_path)
    feedback = Feedback(db_path.parent / "feedback.jsonl")
    collections = Collections(db_path.parent / "collections")
    moods = Moods(db_path.parent / "moods.json")

    def matrix_for(request):
        return library.matrices().get(request.query_params.get("model", ""))

    def heard(path: str) -> str:
        return library.heard_from(Path(path))[0]

    def limit(request) -> int:
        return max(1, min(100, int(request.query_params.get("k", 20))))

    def narrowed(request, matrix):
        """The sound categories of a matrix's files, and which files pass the request's filters.

        Filters: `min` and `max` length in seconds, and any number of `f=kind:value`
        (a file must match one value of every kind that is given).
        """
        sounds, index = library.filters(matrix)
        params = request.query_params
        keep = np.ones(len(matrix.paths), dtype=bool)
        if "min" in params or "max" in params:
            lengths = np.array(matrix.durations)
            keep &= (lengths >= float(params.get("min", 0))) & (lengths < float(params.get("max", "inf")))
        picks: dict[str, set[str]] = {}
        for item in params.getlist("f"):
            kind, _, value = item.partition(":")
            picks.setdefault(kind, set()).add(value)
        if picks:
            keep &= facets.mask(index, len(matrix.paths), picks)
        return sounds, keep

    def steered(request, matrix, vector, asked, keep):
        """With `steer` in the request, bend the search by the user's votes on it.

        The query moves toward the files they liked and away from the ones they
        rejected, and the rejected files themselves are left out.
        """
        if "steer" not in request.query_params:
            return vector, keep
        means, where = library.vectors(matrix)
        votes = {where[path]: verdict for path, verdict in feedback.votes(asked).items() if path in where}
        if not votes:
            return vector, keep
        liked = np.array([i for i, verdict in votes.items() if verdict == 1], dtype=int)
        rejected = np.array([i for i, verdict in votes.items() if verdict == -1], dtype=int)
        keep = keep.copy()
        keep[rejected] = False
        return search.steer(vector, means[liked], means[rejected], STEER_WEIGHT), keep

    def listed(matrix, model, positions):
        """Result rows for files picked by position rather than by a search."""
        sounds, _ = library.filters(matrix)
        hits = [search.Hit(matrix.paths[i], 0.0, 0.0, i, matrix.tags[i]) for i in positions]
        return hits_json(matrix, model, hits, sounds=sounds, root=library.root(matrix), heard=heard)

    def page(request):
        return FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    def library_info(request):
        matrices = library.matrices()
        every_path = [p for m in matrices.values() for p in m.paths]
        # Filters are offered from the model that covers the most files, once CLAP has loaded.
        widest = max(matrices.values(), key=lambda m: len(m.paths), default=None)
        offered = None
        if widest is not None and (classify.MODEL not in matrices or library.ready(classify.MODEL)):
            offered = facets.summary(library.filters(widest)[1])
        return JSONResponse({
            "filters": offered,
            "root": common_root(every_path) if every_path else "",
            "files": len(set(every_path)),
            "renders": {
                "folder": str(library.renders.root) if library.renders.root else None,
                "connected": library.renders.connected(),
            },
            "models": [
                {"name": name, "label": models.EMBEDDERS[name].label, "files": len(m.paths)}
                for name, m in matrices.items()
            ],
        })

    def text_search(request):
        matrix = matrix_for(request)
        query = request.query_params.get("q", "").strip()
        if matrix is None or not query:
            return error("Pick a model and type a description.", 400)
        model = request.query_params["model"]
        vector = library.text_vector(model, query)
        asked = {"kind": "text", "text": query}
        sounds, keep = narrowed(request, matrix)
        vector, keep = steered(request, matrix, vector, asked, keep)
        hits = search.rank(matrix, vector, limit(request), keep=keep)
        return JSONResponse(
            hits_json(matrix, model, hits, lambda p: feedback.verdict(asked, p), sounds, library.root(matrix), heard)
        )

    def similar(request):
        matrix = matrix_for(request)
        if matrix is None:
            return error("Unknown model.", 400)
        path = library.path_of(int(request.query_params.get("id", -1)))
        vector = search.file_vector(matrix, str(path)) if path else None
        if vector is None:
            return JSONResponse([])
        model = request.query_params["model"]
        asked = {"kind": "like", "path": str(path)}
        sounds, keep = narrowed(request, matrix)
        vector, keep = steered(request, matrix, vector, asked, keep)
        hits = search.rank(matrix, vector, limit(request), exclude=str(path), keep=keep)
        return JSONResponse(
            hits_json(matrix, model, hits, lambda p: feedback.verdict(asked, p), sounds, library.root(matrix), heard)
        )

    async def vote(request):
        """Save a thumbs up (1), thumbs down (-1) or cleared vote (0) on one result of one search."""
        body = await request.json()
        asked, verdict = body.get("query") or {}, body.get("verdict")
        path = library.path_of(int(body.get("id", -1)))
        if path is None or verdict not in (-1, 0, 1):
            return error("That vote could not be saved.", 400)
        if asked.get("kind") == "text" and str(asked.get("text", "")).strip():
            asked = {"kind": "text", "text": asked["text"].strip()}
        else:
            seed = library.path_of(int(asked.get("id", -1))) if asked.get("kind") == "like" else None
            if seed is None:
                return error("That vote could not be saved.", 400)
            asked = {"kind": "like", "path": str(seed)}
        shown = {k: body[k] for k in ("model", "rank", "score", "start_s", "window_s") if k in body}
        feedback.record(asked, str(path), verdict, **shown)
        return JSONResponse({"vote": verdict})

    def collection_list(request):
        return JSONResponse([{"name": name, "count": count} for name, count in collections.names()])

    def collection_files(request):
        name = clean_name(request.path_params["name"])
        widest = library.widest()
        if name is None or widest is None:
            return error("No such collection.", 404)
        model, matrix = widest
        _, where = library.vectors(matrix)
        paths = collections.paths(name)
        positions = [where[path] for path in paths if path in where]
        return JSONResponse({
            "hits": listed(matrix, model, positions),
            "missing": len(paths) - len(positions),
            "folder": str(collections.folder(name)),
        })

    async def collection_change(request):
        """Put a file in a collection ({"id": ..., "keep": true}) or take it out (false)."""
        body = await request.json()
        name = clean_name(request.path_params["name"])
        path = library.path_of(int(body.get("id", -1)))
        if name is None:
            return error("A collection name cannot be empty, start with a dot, or contain / : or \\.", 400)
        if path is None:
            return error("That file is not in the index.", 400)
        if body.get("keep"):
            collections.add(name, path)
        else:
            collections.remove(name, path)
        return JSONResponse({"count": len(collections.paths(name))})

    def collection_reveal(request):
        name = clean_name(request.path_params["name"])
        if name is None or not collections.folder(name).is_dir():
            return error("That collection has no folder yet. Keep a file in it first.", 404)
        subprocess.run(["open", str(collections.folder(name))], check=False)
        return JSONResponse({"revealed": str(collections.folder(name))})

    def duplicate_files(request):
        """Sets of byte-identical files, the ones wasting the most space first."""
        widest = library.widest()
        if widest is None:
            return JSONResponse({"sets": [], "set_count": 0, "spare_bytes": 0})
        model, matrix = widest
        found = library.duplicates(matrix)
        return JSONResponse({
            "sets": [
                {"bytes_each": size, "hits": listed(matrix, model, positions)}
                for positions, size in found[:DUPLICATE_SETS_SHOWN]
            ],
            "set_count": len(found),
            "spare_files": sum(len(positions) - 1 for positions, _ in found),
            "spare_bytes": sum(size * (len(positions) - 1) for positions, size in found),
        })

    def mood_list(request):
        return JSONResponse(moods.all())

    async def mood_save(request):
        body = await request.json()
        name, description = clean_name(body.get("name", "")), " ".join(str(body.get("description", "")).split())
        if name is None or not description:
            return error("A mood needs a short name and a description.", 400)
        moods.save(name, description)
        return JSONResponse(moods.all())

    def mood_remove(request):
        moods.remove(request.path_params["name"])
        return JSONResponse(moods.all())

    def located(request):
        file_id = int(request.path_params["file_id"])
        path = library.path_of(file_id)
        return file_id, path if path and path.is_file() else None

    def peaks(request):
        file_id, path = located(request)
        if path is None:
            return error("That file is no longer on disk.", 404)
        try:
            levels = library.peaks(path)
        except RuntimeError as e:
            return error(str(e), 500)
        if levels is None:
            return error("That file is online-only and no waveform was stored for it.", 404)
        return JSONResponse(levels)

    def stored_peaks(request):
        """The stored outlines of several files at once, as {id: levels}.

        Files without a stored outline are left out; the page then asks for those one by one.
        """
        out = {}
        for item in request.query_params.get("ids", "").split(",")[:OUTLINES_PER_REQUEST]:
            path = library.path_of(int(item)) if item.isdigit() else None
            if path is not None and path.is_file():
                levels = library.peaks(path, build=False)
                if levels is not None:
                    out[item] = levels
        return JSONResponse(out)

    def listen(request):
        """The original file when a browser can play it, otherwise a converted copy.

        The page asks again with ?converted=1 if the browser rejects the original.
        """
        file_id, path = located(request)
        if path is None:
            return error("That file is no longer on disk.", 404)
        source, file = library.heard_from(path)
        if source == "none":
            return error("The original is online-only and its render is not within reach.", 404)
        try:
            # A render is either the AAC copy or, where that would be larger, the original's own bytes.
            media_type = BROWSER_PLAYABLE.get(file.suffix.lower())
            direct = (
                "converted" not in request.query_params
                and media_type is not None
                and audio.channel_count(file) <= 2
            )
            if direct:
                return FileResponse(file, media_type=media_type)
            if source == "original" and file.suffix.lower() != renders.AAC:
                # The AAC render is the same conversion a preview is, so one that is current saves making it.
                stat = file.stat()
                copy = library.renders.copy_of(str(file), (stat.st_size, stat.st_mtime))
                if copy is not None and copy.suffix == renders.AAC:
                    return FileResponse(copy, media_type="audio/mp4")
            return FileResponse(library.preview(file_id, file), media_type="audio/mp4")
        except RuntimeError as e:
            return error(str(e), 500)

    def reveal(request):
        _, path = located(request)
        if path is None:
            return error("That file is no longer on disk.", 404)
        subprocess.run(["open", "-R", str(path)], check=False)
        return JSONResponse({"revealed": str(path)})

    def warm_up():
        for name in library.matrices():
            library.embedder(name)
        for matrix in library.matrices().values():
            library.filters(matrix)

    threading.Thread(target=warm_up, daemon=True).start()

    return Starlette(routes=[
        Route("/", page),
        Route("/api/library", library_info),
        Route("/api/search", text_search),
        Route("/api/similar", similar),
        Route("/api/peaks/{file_id:int}", peaks),
        Route("/api/outlines", stored_peaks),
        Route("/audio/{file_id:int}", listen),
        Route("/api/reveal/{file_id:int}", reveal, methods=["POST"]),
        Route("/api/feedback", vote, methods=["POST"]),
        Route("/api/collections", collection_list),
        Route("/api/collections/{name}", collection_files),
        Route("/api/collections/{name}", collection_change, methods=["POST"]),
        Route("/api/collections/{name}/reveal", collection_reveal, methods=["POST"]),
        Route("/api/duplicates", duplicate_files),
        Route("/api/moods", mood_list),
        Route("/api/moods", mood_save, methods=["POST"]),
        Route("/api/moods/{name}", mood_remove, methods=["DELETE"]),
    ])
