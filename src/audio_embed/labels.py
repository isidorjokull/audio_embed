"""Classifiers read off a file without listening to it.

A label is a (kind, value) pair such as ("bpm", "82") or ("category", "kick").
They come from the file's name and folders, its embedded metadata and codec, and an
optional list of folders that say what everything under them is.
"""

import json
import re
import subprocess
import unicodedata
from pathlib import Path

from .audio import is_local

Labels = set[tuple[str, str]]

# Category -> the words that mean it in a file or folder name, English and Icelandic.
CATEGORIES = {
    "kick": "kick kicks kik bassdrum",
    "snare": "snare snares rimshot",
    "clap": "clap claps",
    "hat": "hat hats hihat hihats",
    "tom": "tom toms",
    "cymbal": "cymbal cymbals",
    "percussion": "perc percs percussion conga congas bongo bongos shaker shakers tambourine cowbell",
    "808": "808 808s",
    "breakbeat": "breaks breakbeat breakbeats",
    "drums": "drum drums drm drms trommur",
    "fill": "fill fills",
    "bass": "bass bassi bassa",
    "synth": "synth synths modular",
    "pad": "pad pads",
    "piano": "piano píanó",
    "guitar": "guitar guitars gtr gítar",
    "cello": "cello sello selló celló",
    "strings": "string strings strengir strengja violin viola víóla víólu",
    "choir": "choir kór",
    "vocal": "vocal vocals vox voice rödd",
    "speech": "speech dialogue tal",
    "organ": "organ orgel",
    "bell": "bell bells klukka klukku",
    "chord": "chord chords hljómar hljómur",
    "drone": "drone drones dróni drónar",
    "impact": "impact impacts",
    "hit": "hit hits",
    "whoosh": "whoosh whooshes swoosh swish",
    "riser": "riser risers",
    "texture": "texture textures áferð áferðir",
    "ambience": "ambience ambiance ambient amb atmo atmó atmos atmosphere",
    "noise": "noise",
    "glitch": "glitch glitches",
    "crackle": "crackle crackles brak",
    "rumble": "rumble drunur",
    "granular": "granular",
    "fx": "fx sfx efx",
    "water": "water underwater vatn stream",
    "rain": "rain rigning",
    "fire": "fire eldur",
    "wind": "wind vindur",
    "thunder": "thunder",
    "snow": "snow ice",
    "door": "door doors doorbell doorbells hurð",
    "footsteps": "footstep footsteps fótatak",
    "breath": "breath breathing andardráttur",
    "animal": "animal animals bird birds dýr fugl fuglar",
    "crowd": "crowd applause",
    "vehicle": "car cars train trains traffic strætó",
    "phone": "phone phones",
    "electricity": "electricity electric rafmagn",
    "machine": "machine machines industrial motor",
    "wood": "wood viður",
    "metal": "metal",
    "glass": "glass",
    "stone": "stone stones",
    "explosion": "explosion explosions",
}
WORDS = {word: category for category, words in CATEGORIES.items() for word in words.split()}

# The kinds read from inside a file. They are kept for a file whose contents are not at hand.
FROM_CONTENTS = {"genre", "artist", "encoded by", "year", "bpm", "description", "quality"}

# Codecs that keep every sample, besides plain PCM. Anything else has thrown some of the sound away.
LOSSLESS_CODECS = {"flac", "alac", "ape", "wavpack", "tta", "mlp", "truehd", "shorten", "mp4als", "wmalossless"}
# What an extension usually holds, for a file whose codec was never read.
LOSSY_EXTENSIONS = {".mp3", ".m4a", ".ogg", ".opus"}

# How many folders above the file are read for category and loop words.
NEARBY_FOLDERS = 4
BPM_RANGE = (60, 200)

EXPLICIT_BPM = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d)?)\s*bpm|bpm[\s_-]*(\d{2,3}(?:\.\d)?)", re.I)
NUMBER = re.compile(r"(?<![\d.])\d{2,3}(?:\.\d)?(?![\d.]*\d)")
ONE_SHOT = re.compile(r"one[\s_-]?shots?", re.I)
# A key needs an accidental or a mode to count ("C#", "Em", "Bbmin"): a bare
# letter is too often a take name. A bare letter in square brackets does count.
KEY = re.compile(r"([A-G])([#b]?)(m|min|minor|maj|major)?")
BOILERPLATE = re.compile(
    r"www\.|https?:|@|\(c\)|©|copyright|distributed|all rights|(designed|modified|recorded|made) (by|on|with)", re.I
)
# Where "PercLoop" and "808hat" fall apart into words.
JOINS = re.compile(r"(?<=[a-zß-ÿ])(?=[A-ZÀ-Þ])|(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
MAX_DESCRIPTION = 300


def nfc(text: str) -> str:
    """macOS hands back decomposed accents; compose them so "ó" is one letter."""
    return unicodedata.normalize("NFC", text)


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", JOINS.sub(" ", nfc(text)).casefold())


def number(text: str) -> str:
    return text.removesuffix(".0")


def _tempo(parts: list[str], is_loop: bool) -> str | None:
    """`parts` are the file name, then its folders from nearest to farthest."""
    for part in parts:
        if match := EXPLICIT_BPM.search(part):
            return number(match[1] or match[2])
    if not is_loop:
        return None
    # A loop's tempo is usually just a number in its name or its folder's name.
    for part in parts[:2]:
        low, high = BPM_RANGE
        found = {number(n) for n in NUMBER.findall(part) if low <= float(n) <= high}
        if len(found) == 1:
            return found.pop()
    return None


def _keys(stem: str) -> set[str]:
    found = set()
    for token in re.split(r"[\s_\-().,]+", stem):
        bracketed = token.startswith("[") and token.endswith("]")
        match = KEY.fullmatch(token.strip("[]"))
        if match and (bracketed or match[2] or match[3]):
            minor = match[3] in ("m", "min", "minor")
            found.add(match[1] + match[2] + ("m" if minor else ""))
    return found


def from_path(path: Path) -> Labels:
    """Tempo, key, loop or one-shot, category and folder, from the name and folders alone."""
    stem = nfc(path.stem)
    folders = [nfc(p) for p in path.parent.parts[-NEARBY_FOLDERS:]][::-1]
    out: Labels = set()
    if path.parent.name:
        out.add(("folder", nfc(path.parent.name)))

    kind = None
    for text in (stem, " ".join(folders)):
        if ONE_SHOT.search(text):
            kind = "one-shot"
        elif {"loop", "loops"} & set(words(text)):
            kind = "loop"
        if kind:
            out.add(("type", kind))
            break

    if tempo := _tempo([stem, *folders], kind == "loop"):
        out.add(("bpm", tempo))
    out |= {("key", key) for key in _keys(stem)}

    for word in words(" ".join([stem, *folders])):
        if word in WORDS:
            out.add(("category", WORDS[word]))
    return out


def from_metadata(tags: dict[str, str]) -> Labels:
    """Labels from embedded tags (ID3, RIFF INFO, Broadcast Wave), keyed in lower case."""
    out: Labels = set()
    for kind, key in (("genre", "genre"), ("artist", "artist"), ("encoded by", "encoded_by")):
        if value := tags.get(key, "").strip():
            out.add((kind, value))
    if year := re.match(r"(19|20)\d\d", tags.get("date", "")):
        out.add(("year", year[0]))
    try:
        tempo = float(tags.get("tbpm", 0))
    except ValueError:
        tempo = 0
    if tempo > 0:
        out.add(("bpm", number(f"{tempo:.1f}")))
    # Sound libraries put a sentence about the sound here; others put an advert.
    text = " ".join((tags.get("description") or tags.get("comment") or "").split())
    if len(text.split()) >= 3 and not BOILERPLATE.search(text):
        out.add(("description", text[:MAX_DESCRIPTION]))
    return out


def quality(codec: str, suffix: str) -> str:
    """"lossless" or "lossy", by the codec of the audio, or by the extension when that is not known.

    The extension alone is not enough: an .m4a holds AAC or Apple Lossless, and a
    .wav can hold compressed audio.
    """
    if not codec:
        return "lossy" if suffix.lower() in LOSSY_EXTENSIONS else "lossless"
    return "lossless" if codec.startswith("pcm_") or codec in LOSSLESS_CODECS else "lossy"


def from_location(path: Path, locations: dict[str, dict[str, str]]) -> Labels:
    """Labels given to everything under a folder.

    `locations` maps a folder to {kind: value}. A value may name a folder below
    it by depth: "{0}" is the first folder under the location, "{1}" the next.
    Where locations nest, the deeper one wins for a kind they both set.
    """
    target = nfc(str(path.parent)) + "/"
    found: dict[str, str] = {}
    for folder in sorted(locations, key=len):
        root = nfc(folder).rstrip("/") + "/"
        if not target.startswith(root):
            continue
        below = [p for p in target[len(root):].split("/") if p]
        for kind, value in locations[folder].items():
            try:
                found[kind] = value.format(*below)
            except IndexError:
                found.pop(kind, None)  # the file sits above the folder the value names
    return set(found.items())


def load_locations(path: Path) -> dict[str, dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def probe(path: Path) -> tuple[dict[str, str], str]:
    """The file's embedded tags and the codec of its audio, or nothing if it cannot be read."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags:stream=codec_name,codec_type",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        probed = json.loads(out.stdout)
    except json.JSONDecodeError:
        return {}, ""
    tags = dict(probed.get("format", {}).get("tags", {}))
    codec = ""
    for stream in probed.get("streams", []):
        tags.update(stream.get("tags", {}))
        # Cover art is a stream too, so take the codec of the first one that is audio.
        if not codec and stream.get("codec_type") == "audio":
            codec = stream.get("codec_name", "")
    return {key.lower(): str(value) for key, value in tags.items()}, codec


def read(path: Path, locations: dict[str, dict[str, str]], earlier: Labels | None = None) -> Labels:
    """Every label for one file.

    `earlier` is what the file was labelled with before. With it, a file that is
    online-only (so its contents cannot be read) keeps the labels its contents once gave.
    """
    out = from_path(path) | from_location(path, locations)
    if earlier is not None and not is_local(path):
        kept = {pair for pair in earlier if pair[0] in FROM_CONTENTS}
        if not any(kind == "quality" for kind, _ in kept):
            kept.add(("quality", quality("", path.suffix)))
        return out | kept
    tags, codec = probe(path)
    return out | from_metadata(tags) | {("quality", quality(codec, path.suffix))}
