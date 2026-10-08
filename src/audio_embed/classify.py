"""Sound-type labels for files, assigned by CLAP from text descriptions.

Each category is a name and a description of how it sounds. A file gets the
category whose description its audio is closest to. Nothing is trained and
nothing is stored: labels are worked out from the index when the server starts,
so editing the list only needs a restart.

To use your own list, put a JSON file next to the index (data/categories.json):
    [{"name": "Drone", "description": "a sustained low drone"}, ...]
"""

import json
from pathlib import Path

import numpy as np

# The model the labels come from. Its text and audio vectors share one space.
MODEL = "clap"

DEFAULT_CATEGORIES = [
    ("Drone", "a sustained low drone"),
    ("Ambience", "the background ambience of a place"),
    ("Texture", "an abstract evolving noise texture"),
    ("Impact", "a single heavy impact or hit"),
    ("Whoosh", "a whoosh or swish passing by"),
    ("Water", "water flowing, splashing or dripping"),
    ("Wind and weather", "wind, rain or thunder"),
    ("Crackle", "crackling, popping or clicking noise"),
    ("Electricity", "electrical buzz, hum or sparks"),
    ("Machine", "a machine or engine running"),
    ("Vehicle", "a car, train or other vehicle"),
    ("Footsteps and movement", "footsteps or clothing movement"),
    ("Objects", "handling, dropping or scraping small objects"),
    ("Voice", "a human voice speaking, screaming or singing"),
    ("Crowd", "a crowd of people"),
    ("Animals", "birds, insects or other animals"),
    ("Drums", "drums or percussion"),
    ("Music", "music played on instruments"),
    ("Synth", "a synthesizer tone or musical note"),
]


def load_categories(path: Path) -> list[tuple[str, str]]:
    if path.exists():
        return [(c["name"], c["description"]) for c in json.loads(path.read_text(encoding="utf-8"))]
    return DEFAULT_CATEGORIES


def assign(file_vectors: np.ndarray, category_vectors: np.ndarray) -> np.ndarray:
    """For each file, the index of the category it is closest to."""
    return (file_vectors @ category_vectors.T).argmax(axis=1)
