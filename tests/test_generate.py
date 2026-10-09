import json
import shutil
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from audio_embed import generate, splice
from audio_embed.generate import Ask


def write_wav(path, seconds=1.0, sr=8000, channels=1):
    tone = np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)
    frames = np.repeat((tone * 16000).astype("<i2")[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(frames.tobytes())


def tags(path):
    """The tags embedded in a file, read with ffprobe."""
    import subprocess

    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return {key.lower(): value for key, value in json.loads(out.stdout).get("format", {}).get("tags", {}).items()}


def stand_in(calls=None, fails=False, short=0):
    """Stable Audio 3 as far as this tool can tell: a program that writes --out from --init-audio."""

    def run(command, **how):
        if calls is not None:
            calls.append((command, how))
        if fails:
            return SimpleNamespace(returncode=1, stdout="", stderr="Traceback\nValueError: no such model\n")
        out = command[command.index("--out") + 1]
        seconds = float(command[command.index("--seconds") + 1])
        if "--inpaint-range" in command:  # a level no sample has, as long as was asked for (or `short` samples less)
            splice.write(out, np.full((int(round(seconds * 44100)) - short, 2), 9000, dtype="<i2"))
        elif "--init-audio" in command:
            shutil.copy(command[command.index("--init-audio") + 1], out)
        else:  # from text alone: a clip of the asked length
            write_wav(out, seconds=seconds, sr=44100, channels=2)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def checkout(tmp_path, weights=generate.WEIGHTS["medium"]):
    """A folder that looks like a Stable Audio 3 checkout with these weight files in place."""
    sa3 = tmp_path / "sa3"
    for file in [".venv/bin/python", "scripts/sa3_mlx.py", *(f"models/mlx/{name}" for name in weights)]:
        (sa3 / file).parent.mkdir(parents=True, exist_ok=True)
        (sa3 / file).touch()
    return sa3


def generator(tmp_path, **changes):
    kept = tmp_path / "lib" / "Generated"
    kept.parent.mkdir(exist_ok=True)
    generate.setup(kept)
    how = {"cache": tmp_path / "data" / "generating", "sa3": checkout(tmp_path), "keep_root": kept, "run": stand_in()}
    return generate.Generator(**{**how, **changes})


def made(gen, source, ask=Ask(count=1), start_s=0.0, duration=2.0):
    """Start a run and wait for it; gives its clips as the page would see them."""
    run = gen.start(source, key=str(source), name=source.stem, duration=duration, start_s=start_s, ask=ask)
    for _ in range(500):
        progress = gen.progress(run)
        if progress["done"]:
            return progress["clips"]
        time.sleep(0.01)
    raise AssertionError("the run never finished")


@pytest.mark.parametrize("duration, matched_at, asked, want", [
    (11.3, 0.0, None, (0.0, 11.3)),       # a short sample is used whole
    (600.0, 200.0, None, (200.0, 60.0)),  # a long one from the matched passage, a minute of it
    (100.0, 90.0, None, (40.0, 60.0)),    # pulled back so the minute fits before the end
    (600.0, 200.0, 20.0, (200.0, 20.0)),  # an asked-for length
    (11.3, 0.0, 20.0, (0.0, 11.3)),       # never longer than the sample
])
def test_passage_is_the_sample_or_a_minute_from_where_it_matched(duration, matched_at, asked, want):
    assert generate.passage(duration, matched_at, asked) == want


def test_command_runs_the_checkout_with_its_own_python():
    ask = Ask(prompt="slow metallic shimmer", distance="far", model="sfx")
    command = generate.command(Path("/sa3"), ask, Path("/c/ref.wav"), 11.3, 7, Path("/c/out.wav"))
    assert command == [
        "/sa3/.venv/bin/python", "/sa3/scripts/sa3_mlx.py",
        "--prompt", "slow metallic shimmer", "--dit", "sm-sfx", "--decoder", "same-s",
        "--init-audio", "/c/ref.wav", "--init-noise-level", "0.8",
        "--seconds", "11.3", "--seed", "7", "--out", "/c/out.wav",
    ]


def test_command_passes_prompt_strength_and_what_to_avoid():
    ask = Ask(prompt="pad", avoid="drums, vocals", strength=3.0)
    command = generate.command(Path("/sa3"), ask, Path("/c/ref.wav"), 30.0, 1, Path("/c/out.wav"))
    assert command[-4:] == ["--cfg", "3.0", "--negative-prompt", "drums, vocals"]


def test_a_request_from_the_page_is_checked_and_given_defaults():
    assert generate.ask_from({}) == Ask(prompt="", avoid="", distance="medium", strength=1.0, count=4, seconds=None, model="medium")
    ask = generate.ask_from({"prompt": "  slow   shimmer ", "distance": "close", "count": 2, "seconds": 20, "model": "music", "strength": 2})
    assert ask == Ask(prompt="slow shimmer", distance="close", strength=2.0, count=2, seconds=20.0, model="music")
    # What to avoid does nothing at strength 1, so asking for it raises the strength.
    assert generate.ask_from({"avoid": "drums"}).strength == 3.0
    assert generate.ask_from({"avoid": "drums", "strength": 5}).strength == 5.0


@pytest.mark.parametrize("body", [
    {"distance": "nearby"}, {"model": "large"}, {"count": 0}, {"count": 9}, {"count": "many"},
    {"seconds": 0.2}, {"seconds": 500}, {"strength": -1}, {"strength": 11},
])
def test_a_request_outside_what_the_panel_offers_is_refused(body):
    with pytest.raises(ValueError):
        generate.ask_from(body)


def test_kept_name_says_the_sample_and_the_prompt_and_is_never_taken():
    assert generate.kept_name("Cmin 7th 3", "slow metallic shimmer", set()) == "Cmin 7th 3 - slow metallic shimmer 1.wav"
    taken = {"Cmin 7th 3 - slow metallic shimmer 1.wav", "cmin 7th 3 - slow metallic shimmer 2.WAV"}
    assert generate.kept_name("Cmin 7th 3", "slow metallic shimmer", taken) == "Cmin 7th 3 - slow metallic shimmer 3.wav"
    assert generate.kept_name("Cmin 7th 3", "", set()) == "Cmin 7th 3 - variation 1.wav"
    # Nothing a file name cannot hold, and no prompt so long that the name is unreadable.
    assert generate.kept_name("a", "rain / wind: heavy\\storm?", set()) == "a - rain wind heavy storm 1.wav"
    long = generate.kept_name("a", "one two three four five six seven eight nine ten eleven twelve thirteen", set())
    assert long == "a - one two three four five six seven eight 1.wav"


def test_description_says_where_a_clip_came_from():
    told = generate.describe({
        "name": "Cmin 7th 3", "start_s": 75.0, "seconds": 60.0, "prompt": "slow shimmer", "avoid": "drums",
        "distance": "far", "strength": 3.0, "model": "medium", "seed": 7,
    })
    assert told == (
        'Generated from "Cmin 7th 3" (1:15 to 2:15). Prompt: slow shimmer. Avoid: drums.'
        " Distance: far. Prompt strength: 3. Model: Stable Audio 3 medium. Seed: 7."
    )
    plain = generate.describe({
        "name": "a", "start_s": 0.0, "seconds": 2.0, "prompt": "", "avoid": "",
        "distance": "medium", "strength": 1.0, "model": "sfx", "seed": 1,
    })
    assert plain == 'Generated from "a". Distance: medium. Model: Stable Audio 3 sfx. Seed: 1.'


def test_cut_makes_the_reference_the_generator_reads(tmp_path):
    import subprocess

    plain = tmp_path / "plain.wav"
    write_wav(plain, seconds=3.0)
    source = tmp_path / "long.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(plain), "-c", "copy", "-metadata", "comment=from a sample pack", str(source)], check=True)
    assert tags(source)["comment"] == "from a sample pack"
    out = tmp_path / "ref.wav"
    generate.cut(source, 1.0, 1.5, out)
    with wave.open(str(out)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (44100, 2, 2)
        assert f.getnframes() == pytest.approx(1.5 * 44100, abs=50)
    assert "comment" not in tags(out)  # the sample's own tags are not passed on to its clips


def test_a_run_makes_its_clips_one_per_seed_from_the_asked_passage(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=2.0)
    clips = made(gen, source, Ask(prompt="shimmer", count=2), start_s=0.5, duration=2.0)
    assert [clip["state"] for clip in clips] == ["done", "done"]
    assert [clip["seconds"] for clip in clips] == [2.0, 2.0]

    commands = [command for command, _ in calls]
    seeds = [command[command.index("--seed") + 1] for command in commands]
    assert len(set(seeds)) == 2
    assert all(how["cwd"] == gen.sa3 for _, how in calls)
    # Every clip of a run is made from the same cut of the sample.
    assert len({command[command.index("--init-audio") + 1] for command in commands}) == 1
    for clip in clips:
        assert gen.clip(clip["id"]).is_file()
    assert [c["id"] for c in gen.clips_from(str(source))] == [clip["id"] for clip in clips]
    assert gen.clips_from(str(tmp_path / "another.wav")) == []


def test_a_failed_clip_says_why_and_the_run_still_ends(tmp_path):
    gen = generator(tmp_path, run=stand_in(fails=True))
    source = tmp_path / "tone.wav"
    write_wav(source)
    (clip,) = made(gen, source)
    assert clip["state"] == "failed"
    assert clip["error"] == "ValueError: no such model"
    assert gen.clips_from(str(source)) == []


def test_a_clip_id_is_never_a_way_to_another_file(tmp_path):
    gen = generator(tmp_path)
    source = tmp_path / "data" / "secret.wav"  # a file next to the cache folder
    source.parent.mkdir(parents=True, exist_ok=True)
    write_wav(source)
    (clip,) = made(gen, source)
    assert gen.clip(clip["id"]) is not None
    for bad in ("../secret", "..%2Fsecret", "", "secret", clip["id"] + "/../../secret", clip["id"].upper() + "0"):
        assert gen.clip(bad) is None
        assert gen.story(bad) is None


def test_keeping_copies_a_clip_into_the_marked_folder_with_its_story(tmp_path):
    from audio_embed import labels

    gen = generator(tmp_path)
    source = tmp_path / "Cmin 7th.wav"
    write_wav(source, seconds=1.0)
    first, second = made(gen, source, Ask(prompt="slow shimmer", count=2), duration=1.0)

    kept = gen.keep(first["id"])
    assert kept == gen.keep_root / "Cmin 7th - slow shimmer 1.wav"
    assert kept.is_file()
    assert gen.keep(first["id"]) == kept  # keeping it again is not a second copy
    assert gen.keep(second["id"]).name == "Cmin 7th - slow shimmer 2.wav"
    assert sorted(p.name for p in gen.keep_root.glob("*.wav")) == ["Cmin 7th - slow shimmer 1.wav", "Cmin 7th - slow shimmer 2.wav"]

    described = {value for kind, value in labels.read(kept, {}) if kind == "description"}
    assert len(described) == 1 and described.pop().startswith('Generated from "Cmin 7th". Prompt: slow shimmer.')
    assert [c["kept"] for c in gen.clips_from(str(source))] == [str(kept), str(gen.keep_root / "Cmin 7th - slow shimmer 2.wav")]


def test_nothing_is_kept_in_a_folder_without_the_marker(tmp_path):
    gen = generator(tmp_path)
    source = tmp_path / "tone.wav"
    write_wav(source)
    (clip,) = made(gen, source)
    (gen.keep_root / generate.MARKER).unlink()
    with pytest.raises(RuntimeError, match="audio-embed generator"):
        gen.keep(clip["id"])
    assert list(gen.keep_root.iterdir()) == []

    nowhere = generator(tmp_path, keep_root=None)
    (clip,) = made(nowhere, source)
    with pytest.raises(RuntimeError, match="audio-embed generator"):
        nowhere.keep(clip["id"])


def test_the_oldest_clips_go_once_the_cache_is_full(tmp_path):
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=1.0)
    gen = generator(tmp_path, cache_bytes=400_000)  # room for two of these 176 kB clips
    ids = []
    for _ in range(3):
        ids += [clip["id"] for clip in made(gen, source, duration=1.0)]
        time.sleep(0.02)
    assert gen.clip(ids[0]) is None
    assert gen.clip(ids[1]) is not None and gen.clip(ids[2]) is not None
    assert [c["id"] for c in gen.clips_from(str(source))] == ids[1:]
    assert sorted(p.suffix for p in gen.cache.iterdir()) == [".json", ".json", ".wav", ".wav"]


def test_what_is_missing_is_said_before_anything_is_started(tmp_path):
    assert generator(tmp_path).problem("medium") is None
    assert "audio-embed generator" in generator(tmp_path, sa3=None).problem("medium")
    assert "is not a Stable Audio 3 folder" in generator(tmp_path, sa3=tmp_path / "elsewhere").problem("medium")
    # The small models are a separate download.
    assert "dit_sm-sfx_f16.npz" in generator(tmp_path).problem("sfx")
    assert generator(tmp_path).models() == ["medium"]


def test_setup_remembers_both_folders_and_labels_the_kept_one(tmp_path):
    settings = tmp_path / "data" / "settings.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"renders": "/Volumes/Drive/renders"}))
    locations = tmp_path / "data" / "locations.json"
    locations.write_text(json.dumps({"/lib/Foley": {"source": "downloaded sfx"}}))
    kept = tmp_path / "lib" / "Generated"
    kept.parent.mkdir()

    generate.setup(kept)
    generate.save(settings, sa3=checkout(tmp_path), keep_root=kept)
    assert generate.label_folder(locations, kept) is True
    assert generate.label_folder(locations, kept) is False  # already labelled: left as it is

    assert (kept / generate.MARKER).is_file()
    assert json.loads(settings.read_text())["renders"] == "/Volumes/Drive/renders"
    assert generate.saved(settings) == (tmp_path / "sa3", kept)
    assert json.loads(locations.read_text()) == {
        "/lib/Foley": {"source": "downloaded sfx"},
        str(kept): {"source": "generated"},
    }
    with pytest.raises(RuntimeError, match="is not there"):
        generate.setup(tmp_path / "unplugged" / "Generated")


class FakeEmbedder:
    name = "fake"
    sr = 8000
    window_s = 1.0
    batch_size = 2

    def embed_audio(self, windows):
        return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(windows), 1))


def test_taking_in_the_kept_folder_makes_a_new_clip_searchable_at_once(tmp_path):
    from audio_embed.cli import index_folder
    from audio_embed.server import Library
    from audio_embed.store import Store

    lib = tmp_path / "lib"
    lib.mkdir()
    write_wav(lib / "old.wav")
    kept = lib / "Generated"
    generate.setup(kept)
    data = tmp_path / "data"
    data.mkdir()
    (data / "locations.json").write_text(json.dumps({str(kept): {"source": "generated"}}))
    embedder = FakeEmbedder()
    index_folder(Store(data / "index.db"), embedder, lib)

    library = Library(data / "index.db")
    library._embedders["fake"] = embedder  # the model the server would have loaded
    assert [Path(p).name for p in library.matrices()["fake"].paths] == ["old.wav"]

    write_wav(kept / "old - shimmer 1.wav")
    library.take_in(kept)
    matrix = library.matrices()["fake"]  # re-read right away, not at the next spaced-out reload
    assert [Path(p).name for p in matrix.paths] == ["old - shimmer 1.wav", "old.wav"]
    assert matrix.labels[0]["source"] == ["generated"]
    assert library.path_of(matrix.ids[0]) == kept / "old - shimmer 1.wav"


def test_the_generator_command_sets_everything_up_and_says_what_is_ready(tmp_path, capsys, monkeypatch):
    from audio_embed import cli

    sa3 = checkout(tmp_path)
    kept = tmp_path / "lib" / "Generated"
    kept.parent.mkdir()
    db = tmp_path / "data" / "index.db"
    monkeypatch.setattr("sys.argv", ["audio-embed", "--db", str(db), "generator", "--sa3", str(sa3), "--keep-in", str(kept)])
    cli.main()

    assert (kept / generate.MARKER).is_file()
    assert generate.saved(db.parent / "settings.json") == (sa3.resolve(), kept.resolve())
    assert json.loads((db.parent / "locations.json").read_text()) == {str(kept.resolve()): {"source": "generated"}}
    said = capsys.readouterr().out
    assert "medium: ready" in said
    assert "sfx: not ready" in said and "dit_sm-sfx_f16.npz" in said

    # Asked again with nothing new, it only reports.
    monkeypatch.setattr("sys.argv", ["audio-embed", "--db", str(db), "generator"])
    cli.main()
    assert "medium: ready" in capsys.readouterr().out


def finished(gen, run):
    for _ in range(500):
        progress = gen.progress(run)
        if progress["done"]:
            return progress["clips"]
        time.sleep(0.01)
    raise AssertionError("the run never finished")


def test_command_from_text_alone_names_no_sample():
    command = generate.command(Path("/sa3"), Ask(prompt="dark low drone", model="sfx"), None, 10.0, 7, Path("/c/out.wav"))
    assert command == [
        "/sa3/.venv/bin/python", "/sa3/scripts/sa3_mlx.py",
        "--prompt", "dark low drone", "--dit", "sm-sfx", "--decoder", "same-s",
        "--seconds", "10.0", "--seed", "7", "--out", "/c/out.wav",
    ]


def test_a_run_from_text_makes_clips_of_the_asked_length_with_no_sample(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    clips = finished(gen, gen.start_from_text(Ask(prompt="dark low drone", count=2, seconds=3.0)))
    assert [(clip["state"], clip["seconds"]) for clip in clips] == [("done", 3.0), ("done", 3.0)]
    assert all("--init-audio" not in command and "--init-noise-level" not in command for command, _ in calls)
    listed = gen.clips_from(generate.FROM_TEXT)
    assert [(c["id"], c["prompt"]) for c in listed] == [(clip["id"], "dark low drone") for clip in clips]

    # Ten seconds unless a length is asked for.
    (clip,) = finished(gen, gen.start_from_text(Ask(prompt="rain", count=1)))
    assert clip["seconds"] == 10.0


def test_a_run_from_text_needs_a_prompt(tmp_path):
    gen = generator(tmp_path)
    with pytest.raises(ValueError, match="what to generate"):
        gen.start_from_text(Ask(prompt=""))


def test_a_kept_clip_made_from_text_is_named_and_described_by_its_prompt(tmp_path):
    from audio_embed import labels

    gen = generator(tmp_path)
    (clip,) = finished(gen, gen.start_from_text(Ask(prompt="dark low drone", count=1, seconds=1.0, model="medium")))
    kept = gen.keep(clip["id"])
    assert kept.name == "dark low drone 1.wav"
    (described,) = {value for kind, value in labels.read(kept, {}) if kind == "description"}
    seed = gen.story(clip["id"])["seed"]
    assert described == f"Generated from text. Prompt: dark low drone. Model: Stable Audio 3 medium. Seed: {seed}."
    assert generate.kept_name("", "dark low drone", {"Dark Low Drone 1.wav"}) == "dark low drone 2.wav"


def test_a_request_says_what_to_make_and_where():
    ask = generate.ask_from({"make": "part", "span": [3, "4.5"], "prompt": "bells"})
    assert (ask.make, ask.span, ask.prompt) == ("part", (3.0, 4.5), "bells")
    assert generate.ask_from({"make": "loop", "join": "long"}).join == "long"
    longer = generate.ask_from({"make": "longer", "add": 10, "marker": 7.5})
    assert (longer.add, longer.marker) == (10.0, 7.5)
    assert generate.ask_from({"make": "longer"}).marker is None   # the end of the sample
    plain = generate.ask_from({})
    assert (plain.make, plain.span, plain.marker, plain.join, plain.add) == ("variations", None, None, "medium", 30.0)


@pytest.mark.parametrize("body, says", [
    ({"make": "remix"}, "What to make"),
    ({"make": "loop", "join": "huge"}, "The join"),
    ({"make": "part"}, "Mark the part"),
    ({"make": "part", "span": [5, 5]}, "start before it ends"),
    ({"make": "part", "span": [5, 4]}, "start before it ends"),
    ({"make": "part", "span": [-1, 4]}, "start before it ends"),
    ({"make": "part", "span": [1, 1.1]}, "at least 0.2 seconds"),
    ({"make": "part", "span": ["x", 4]}, "must be numbers"),
    ({"make": "part", "span": [4]}, "must be numbers"),
    ({"make": "part", "span": [float("nan"), 4]}, "start before it ends"),
    ({"make": "longer", "add": 0}, "Between 1 and 119"),
    ({"make": "longer", "add": 500}, "Between 1 and 119"),
    ({"make": "longer", "add": float("nan")}, "Between 1 and 119"),
    ({"make": "longer", "marker": 0}, "after the start"),
    ({"make": "longer", "marker": float("nan")}, "after the start"),
])
def test_a_request_for_a_part_a_loop_or_more_is_checked(body, says):
    with pytest.raises(ValueError, match=says):
        generate.ask_from(body)


def test_the_passage_for_a_part_is_placed_around_it():
    short = generate.plan(10.0, 0.0, Ask(make="part", span=(3.0, 5.0)))
    assert (short.start_s, short.seconds, short.whole, short.part, short.out_s) == (0.0, 10.0, True, (3.0, 5.0), 10.0)
    long = generate.plan(600.0, 0.0, Ask(make="part", span=(300.0, 310.0)))
    assert (long.start_s, long.seconds, long.whole) == (275.0, 60.0, False)   # as much before as after
    assert long.part == (25.0, 35.0)
    at_the_end = generate.plan(600.0, 0.0, Ask(make="part", span=(595.0, 600.0)))
    assert (at_the_end.start_s, at_the_end.part) == (540.0, (55.0, 60.0))     # pulled back to fit
    past = generate.plan(10.0, 0.0, Ask(make="part", span=(8.0, 99.0)))
    assert past.part == (8.0, 10.0)                                            # never past the end


@pytest.mark.parametrize("duration, span, says", [
    (10.0, (0.0, 9.5), "Mark a shorter part"),        # under a second would stay
    (600.0, (100.0, 160.0), "Mark a shorter part"),   # as long as the passage itself
    (10.0, (9.9, 12.0), "past the end"),
    (10.0, (20.0, 22.0), "past the end"),
])
def test_a_part_that_leaves_nothing_to_match_is_refused(duration, span, says):
    with pytest.raises(ValueError, match=says):
        generate.plan(duration, 0.0, Ask(make="part", span=span))


def test_a_loop_is_the_passage_variations_would_use_with_a_join_that_fits():
    loop = generate.plan(600.0, 200.0, Ask(make="loop", join="long", seconds=20.0))
    assert (loop.start_s, loop.seconds, loop.join_s, loop.part, loop.out_s) == (200.0, 20.0, 4.0, None, 20.0)
    tight = generate.plan(2.0, 0.0, Ask(make="loop", join="long"))
    assert tight.join_s == 1.0   # never more than half the clip
    with pytest.raises(ValueError, match="at least 2 seconds"):
        generate.plan(1.5, 0.0, Ask(make="loop"))


def test_longer_follows_on_from_the_marker_or_the_end():
    from_the_end = generate.plan(8.0, 0.0, Ask(make="longer", add=10.0))
    assert (from_the_end.start_s, from_the_end.seconds, from_the_end.whole) == (0.0, 8.0, True)
    assert (from_the_end.part, from_the_end.out_s) == ((8.0, 18.0), 18.0)
    pulled_back = generate.plan(8.0, 0.0, Ask(make="longer", add=10.0, marker=6.0))
    assert (pulled_back.seconds, pulled_back.whole, pulled_back.part) == (6.0, False, (6.0, 16.0))
    long = generate.plan(600.0, 0.0, Ask(make="longer", add=30.0))
    assert (long.start_s, long.seconds, long.out_s) == (540.0, 60.0, 90.0)     # the last minute of it
    most = generate.plan(600.0, 0.0, Ask(make="longer", add=100.0))
    assert (most.seconds, most.out_s) == (20.0, 120.0)                         # never more than two minutes in all
    beyond = generate.plan(8.0, 0.0, Ask(make="longer", add=10.0, marker=500.0))
    assert beyond.seconds == 8.0                                               # a marker past the end is the end
    with pytest.raises(ValueError, match="less than 1 second before the marker"):
        generate.plan(8.0, 0.0, Ask(make="longer", marker=0.5))


def test_variations_are_planned_as_before():
    plain = generate.plan(600.0, 200.0, Ask())
    assert (plain.start_s, plain.seconds, plain.whole, plain.part, plain.join_s, plain.out_s) == (200.0, 60.0, False, None, 0.0, 60.0)


def level(n, value=100):
    return np.full((n, 2), value, dtype="<i2")


def test_a_part_is_laid_out_on_steps_inside_the_passage():
    samples = level(10 * 44100)
    given, start, end, total, by = generate.lay_out("part", samples, generate.Plan(0.0, 10.0, True, (3.0, 5.0), 0.0, 10.0))
    assert given is samples and by == 0 and total == len(samples)
    assert (start, end) == (splice.snap(3.0), splice.snap(5.0))
    assert start % splice.STEP == 0 and end % splice.STEP == 0


def test_a_part_that_reaches_the_end_of_a_one_shot_takes_its_whole_tail():
    samples = level(int(3.2 * 44100))   # not a whole number of steps
    _, start, end, total, _ = generate.lay_out("part", samples, generate.Plan(0.0, 3.2, True, (2.5, 3.2), 0.0, 3.2))
    assert end == total == len(samples)   # no sliver of the old tail is left after the part
    made = level(len(samples), 9000)
    out = generate.finish("part", samples, made, start, end, 0)
    assert len(out) == len(samples) and np.array_equal(out[start:], made[start:])


def test_a_loop_is_turned_so_its_ends_meet_inside_the_join():
    samples = np.repeat(np.arange(60 * 4096 + 500, dtype="<i4")[:, None] % 30000, 2, axis=1).astype("<i2")
    given, start, end, total, by = generate.lay_out("loop", samples, generate.Plan(0.0, 5.58, True, None, 2.0, 5.58))
    assert total == len(given) == 60 * 4096      # trimmed to whole steps
    assert by == 30 * 4096
    seam = total - by                              # where the old end meets the old start
    assert start < seam < end and (end - start) == 22 * 4096   # two seconds is 21.5 steps
    assert given[seam - 1].tolist() == samples[total - 1].tolist() and given[seam].tolist() == samples[0].tolist()

    made = level(total, 9000)
    out = generate.finish("loop", given, made, start, end, by)
    assert len(out) == total
    # Turned back: the join is at the two ends, and the middle is the sample as it was.
    assert out[0, 0] == 9000 and out[-1, 0] == 9000
    middle = slice(by - 2 * 4096, by + 2 * 4096)
    assert np.array_equal(out[middle], samples[:total][middle])


def test_a_join_never_takes_more_than_half_a_short_loop():
    samples = level(22 * 4096)   # about two seconds
    _, start, end, total, _ = generate.lay_out("loop", samples, generate.Plan(0.0, 2.04, True, None, 4.0, 2.04))
    assert end - start == 11 * 4096 and 0 < start and end < total


def test_longer_asks_for_the_passage_and_what_follows_it():
    samples = level(8 * 44100)
    given, start, end, total, by = generate.lay_out("longer", samples, generate.Plan(0.0, 8.0, True, (8.0, 18.0), 0.0, 18.0))
    assert len(given) == len(samples) // 4096 * 4096 and by == 0
    assert start == len(given) and end == total == len(given) + splice.snap(10.0)
    out = generate.finish("longer", given, level(total, 9000), start, end, by)
    assert len(out) == total and out[-1, 0] == 9000 and out[0, 0] == 100


def test_command_for_a_part_names_the_stretch_and_not_a_distance():
    sa3 = Path("/sa3")
    cmd = generate.command(sa3, Ask(prompt="bells", make="part"), Path("/tmp/ref.wav"), 10.0, 7, Path("/tmp/out.wav"),
                           part=(12288, 20480, 45056))
    assert cmd[cmd.index("--init-audio") + 1] == "/tmp/ref.wav"
    assert cmd[cmd.index("--inpaint-range") + 1] == "0.278638,0.464398"
    assert cmd[cmd.index("--seconds") + 1] == "1.021677"
    assert "--init-noise-level" not in cmd
    assert cmd[-4:] == ["--seed", "7", "--out", "/tmp/out.wav"]


def story_of(gen, clip):
    return gen.story(clip["id"])


def test_a_redone_part_is_the_sample_with_only_that_part_new(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=4.0, sr=44100, channels=2)
    (clip,) = made(gen, source, Ask(count=1, make="part", span=(1.0, 2.0)), duration=4.0)
    assert clip["state"] == "done"

    command = calls[0][0]
    start, end = splice.snap(1.0), splice.snap(2.0)
    assert command[command.index("--inpaint-range") + 1] == f"{splice.clock(start)},{splice.clock(end)}"
    out, original = splice.read(gen.clip(clip["id"])), splice.read(source)
    assert len(out) == len(original)
    assert np.all(out[start:end] == 9000)
    assert np.array_equal(out[: start - splice.FADE], original[: start - splice.FADE])
    assert np.array_equal(out[end + splice.FADE:], original[end + splice.FADE:])

    story = story_of(gen, clip)
    assert (story["make"], story["join"], story["add"], story["parent"]) == ("part", None, None, None)
    assert story["part"] == [round(start / 44100, 2), round(end / 44100, 2)]
    assert story["length"] == pytest.approx(4.0, abs=0.01) and story["seconds"] == 4.0


def test_a_loop_is_new_at_both_ends_and_untouched_in_the_middle(tmp_path):
    gen = generator(tmp_path)
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=4.0, sr=44100, channels=2)
    (clip,) = made(gen, source, Ask(count=1, make="loop", join="medium"), duration=4.0)
    out, original = splice.read(gen.clip(clip["id"])), splice.read(source)
    assert len(out) == len(original) // 4096 * 4096
    assert out[0, 0] == 9000 and out[-1, 0] == 9000
    middle = slice(len(out) // 2 - 4096, len(out) // 2 + 4096)
    assert np.array_equal(out[middle], original[middle])
    story = story_of(gen, clip)
    assert (story["make"], story["part"]) == ("loop", None)
    assert story["join"] == pytest.approx(2.0, abs=0.1)
    assert clip["seconds"] == pytest.approx(len(out) / 44100)


def test_a_longer_clip_is_the_sample_and_then_new_sound(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=3.0, sr=44100, channels=2)
    (clip,) = made(gen, source, Ask(count=1, make="longer", add=2.0), duration=3.0)
    out, original = splice.read(gen.clip(clip["id"])), splice.read(source)
    kept = len(original) // 4096 * 4096
    assert len(out) == kept + splice.snap(2.0)
    assert np.array_equal(out[: kept - splice.FADE], original[: kept - splice.FADE])
    assert np.all(out[kept:] == 9000)
    command = calls[0][0]
    assert command[command.index("--seconds") + 1] == splice.clock(len(out))
    story = story_of(gen, clip)
    assert story["make"] == "longer" and story["add"] == pytest.approx(2.0, abs=0.1)
    assert clip["seconds"] == pytest.approx(len(out) / 44100)


def test_a_clip_that_comes_back_too_short_fails_and_the_run_goes_on(tmp_path):
    gen = generator(tmp_path, run=stand_in(short=30000))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=3.0, sr=44100, channels=2)
    clips = made(gen, source, Ask(count=2, make="longer", add=1.0), duration=3.0)
    assert [clip["state"] for clip in clips] == ["failed", "failed"]
    assert clips[0]["error"] == "Stable Audio 3 returned a shorter clip than was asked for."
    assert gen.clips_from(str(source)) == []


def test_a_run_that_cannot_be_planned_is_refused_before_it_starts(tmp_path):
    gen = generator(tmp_path)
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=1.0)
    with pytest.raises(ValueError, match="at least 2 seconds"):
        gen.start(source, key=str(source), name="tone", duration=1.0, start_s=0.0, ask=Ask(make="loop"))
    with pytest.raises(ValueError, match="from text alone"):
        gen.start_from_text(Ask(prompt="rain", make="loop"))


BASE = {"name": "Pad 04", "start_s": 0.0, "seconds": 20.0, "whole": True, "prompt": "", "avoid": "",
        "distance": "medium", "strength": 1.0, "model": "medium", "seed": 7}


def test_description_says_what_was_made_anew():
    assert generate.describe({**BASE, "make": "part", "part": [12.0, 15.0], "prompt": "bells"}) == (
        'Generated from "Pad 04". 0:12 to 0:15 regenerated. Prompt: bells. Model: Stable Audio 3 medium. Seed: 7.')
    assert generate.describe({**BASE, "make": "loop", "join": 2.0}) == (
        'Generated from "Pad 04". Made to loop (2 s join). Model: Stable Audio 3 medium. Seed: 7.')
    assert generate.describe({**BASE, "make": "longer", "part": [20.0, 50.0], "add": 30.0}) == (
        'Generated from "Pad 04". Continued for 30 s from 0:20. Model: Stable Audio 3 medium. Seed: 7.')


def test_description_tells_the_whole_chain():
    first = generate.describe({**BASE, "distance": "far", "seed": 3})
    then = generate.describe({**BASE, "make": "loop", "join": 2.0, "seed": 9, "parent": first})
    assert then == ('Generated from "Pad 04". Distance: far. Model: Stable Audio 3 medium. Seed: 3.'
                    " Then made to loop (2 s join). Model: Stable Audio 3 medium. Seed: 9.")
    again = generate.describe({**BASE, "distance": "close", "seed": 4, "parent": then})
    assert again.endswith("Then a variation of it, distance close. Model: Stable Audio 3 medium. Seed: 4.")


def test_a_story_from_before_this_change_is_a_variation():
    assert generate.describe(dict(BASE)) == 'Generated from "Pad 04". Distance: medium. Model: Stable Audio 3 medium. Seed: 7.'


def test_a_run_can_start_from_a_clip_and_is_filed_with_it(tmp_path):
    calls = []
    gen = generator(tmp_path, run=stand_in(calls))
    source = tmp_path / "tone.wav"
    write_wav(source, seconds=4.0, sr=44100, channels=2)
    (first,) = made(gen, source, Ask(count=1, prompt="shimmer"), duration=4.0)

    run = gen.start_from_clip(first["id"], Ask(count=1, make="loop"))
    for _ in range(500):
        if gen.progress(run)["done"]:
            break
        time.sleep(0.01)
    (second,) = gen.progress(run)["clips"]
    assert second["state"] == "done"
    # It was cut from the clip in the cache, not from the library file.
    assert calls[-1][0][calls[-1][0].index("--init-audio") + 1] != str(source)
    story = gen.story(second["id"])
    assert (story["key"], story["name"], story["make"]) == (str(source), "tone", "loop")
    assert story["parent"] == generate.describe(gen.story(first["id"]))
    assert [c["id"] for c in gen.clips_from(str(source))] == [first["id"], second["id"]]
    assert "Then made to loop" in generate.describe(story)


def test_a_clip_that_is_gone_cannot_be_worked_from(tmp_path):
    gen = generator(tmp_path)
    with pytest.raises(ValueError, match="no longer in the cache"):
        gen.start_from_clip("0123456789abcdef", Ask())
    with pytest.raises(ValueError, match="no longer in the cache"):
        gen.start_from_clip("../../etc/passwd", Ask())
