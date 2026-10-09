import wave

import numpy as np
import pytest

from audio_embed import audio, search
from audio_embed.cli import index_folder
from audio_embed.store import Store

SR = 1000


def test_short_file_is_one_span():
    assert audio.window_spans(2500, SR, 10.0) == [(0, 2500)]


def test_exact_multiple_tiles_without_overlap():
    assert audio.window_spans(20_000, SR, 10.0) == [(0, 10_000), (10_000, 20_000)]


def test_tail_gets_a_full_window_ending_at_file_end():
    assert audio.window_spans(25_000, SR, 10.0) == [(0, 10_000), (10_000, 20_000), (15_000, 25_000)]


def test_tiny_tail_is_dropped():
    assert audio.window_spans(20_500, SR, 10.0) == [(0, 10_000), (10_000, 20_000)]


def test_empty_file_has_no_spans():
    assert audio.window_spans(0, SR, 10.0) == []


def test_silence_detection():
    assert audio.is_silent(np.zeros(100, dtype=np.float32))
    assert not audio.is_silent(np.array([0.0, 0.01], dtype=np.float32))


def write_wav(path, channels, seconds=1.0, sr=8000):
    tone = np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)
    frames = np.repeat((tone * 16000).astype("<i2")[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(frames.tobytes())


@pytest.mark.parametrize("channels", [1, 2, 15])
def test_decode_downmixes_any_channel_count(tmp_path, channels):
    path = tmp_path / "tone.wav"
    write_wav(path, channels)
    samples = audio.decode(path, 16_000)
    assert samples.dtype == np.float32
    assert abs(len(samples) - 16_000) < 200
    assert np.abs(samples).max() > 0.05


def test_decode_reports_unreadable_file(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio")
    with pytest.raises(RuntimeError):
        audio.decode(path, 16_000)


def test_peaks_outline_is_scaled_to_the_loudest_point(tmp_path):
    path = tmp_path / "tone.wav"
    write_wav(path, 2, seconds=2.0)
    levels = audio.peaks(path, buckets=50)
    assert len(levels) == 50
    assert max(levels) == 1.0
    assert min(levels) > 0.5  # a steady tone has a flat outline


def test_preview_of_a_many_channel_file_is_stereo(tmp_path):
    source, out = tmp_path / "wide.wav", tmp_path / "wide.m4a"
    write_wav(source, 15)
    audio.write_preview(source, out)
    assert audio.channel_count(out) == 2
    assert np.abs(audio.decode(out, 8000)).max() > 0.05


def unit(*values):
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "index.db")
    s.put("m", "/lib/a.wav", 1, 1.0, 20.0, [(0.0, 10.0), (10.0, 20.0)], [unit(1, 0, 0), unit(0, 1, 0)])
    s.put("m", "/lib/b.wav", 1, 1.0, 5.0, [(0.0, 5.0)], [unit(0, 0, 1)])
    return s


def test_store_round_trip(store):
    matrix = store.load("m")
    assert matrix.paths == ["/lib/a.wav", "/lib/b.wav"]
    assert matrix.file_idx.tolist() == [0, 0, 1]
    assert matrix.starts.tolist() == [0.0, 10.0, 0.0]
    assert matrix.durations == [20.0, 5.0]
    assert len(set(matrix.ids)) == 2
    np.testing.assert_allclose(matrix.vecs[1], unit(0, 1, 0), atol=1e-3)
    assert store.indexed("m") == {"/lib/a.wav": (1, 1.0), "/lib/b.wav": (1, 1.0)}
    assert store.models() == ["m"]


def test_put_replaces_previous_windows(store):
    store.put("m", "/lib/a.wav", 2, 2.0, 5.0, [(0.0, 5.0)], [unit(1, 1, 0)])
    matrix = store.load("m")
    assert matrix.file_idx.tolist() == [0, 1]
    assert store.indexed("m")["/lib/a.wav"] == (2, 2.0)


def test_remove_only_touches_that_model(store):
    store.put("other", "/lib/a.wav", 1, 1.0, 5.0, [(0.0, 5.0)], [unit(1, 0, 0)])
    store.remove("m", ["/lib/a.wav"])
    assert store.load("m").paths == ["/lib/b.wav"]
    assert store.load("other").paths == ["/lib/a.wav"]


def test_tags_belong_to_the_file_and_show_up_on_hits(store):
    store.put("other", "/lib/a.wav", 1, 1.0, 5.0, [(0.0, 5.0)], [unit(1, 0, 0)])
    store.tag(["/lib/a.wav", "/lib/not-indexed.wav"], "foley")
    store.tag(["/lib/a.wav"], "foley")  # tagging twice is harmless
    assert store.load("m").tags == [["foley"], []]
    assert store.load("other").tags == [["foley"]]
    hits = search.rank(store.load("m"), unit(1, 0, 0), k=10)
    assert [h.tags for h in hits] == [["foley"], []]


def test_rank_scores_a_file_by_its_best_window(store):
    hits = search.rank(store.load("m"), unit(0, 1, 0.1), k=10)
    assert [h.path for h in hits] == ["/lib/a.wav", "/lib/b.wav"]
    assert hits[0].start_s == 10.0
    assert hits[0].score > 0.9
    assert [h.index for h in hits] == [0, 1]


def test_rank_can_exclude_the_query_file_and_limit_results(store):
    matrix = store.load("m")
    assert [h.path for h in search.rank(matrix, unit(1, 0, 0), k=10, exclude="/lib/a.wav")] == ["/lib/b.wav"]
    assert len(search.rank(matrix, unit(1, 0, 0), k=1)) == 1


def test_file_vector_is_unit_mean_of_windows(store):
    matrix = store.load("m")
    np.testing.assert_allclose(search.file_vector(matrix, "/lib/a.wav"), unit(1, 1, 0), atol=1e-3)
    assert search.file_vector(matrix, "/lib/missing.wav") is None


class FakeEmbedder:
    name = "fake"
    sr = 8000
    window_s = 1.0
    batch_size = 2

    def __init__(self):
        self.calls = 0

    def embed_audio(self, windows):
        self.calls += len(windows)
        return np.tile(unit(1, 0, 0), (len(windows), 1))


def test_index_is_incremental_and_prunes_deleted_files(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    write_wav(lib / "one.wav", 1, seconds=2.5)
    write_wav(lib / "two.wav", 2, seconds=0.5)
    (lib / "notes.txt").write_text("not audio")
    store = Store(tmp_path / "index.db")
    embedder = FakeEmbedder()

    index_folder(store, embedder, lib)
    assert len(store.load("fake").paths) == 2
    assert embedder.calls == 3  # 2.5 s -> 2 windows (0.5 s tail dropped), 0.5 s -> 1 window

    index_folder(store, embedder, lib)
    assert embedder.calls == 3  # nothing changed, nothing re-embedded

    (lib / "two.wav").unlink()
    index_folder(store, embedder, lib)
    assert store.load("fake").paths == [str(lib / "one.wav")]


def test_a_folder_spelled_two_ways_is_indexed_once(tmp_path):
    import unicodedata

    composed = tmp_path / unicodedata.normalize("NFC", "sýning")
    decomposed = tmp_path / unicodedata.normalize("NFD", "sýning")
    composed.mkdir()
    if not decomposed.is_dir():
        pytest.skip("this file system tells the two spellings apart")
    write_wav(composed / "one.wav", 1, seconds=1.0)
    store = Store(tmp_path / "index.db")
    embedder = FakeEmbedder()

    index_folder(store, embedder, composed)
    index_folder(store, embedder, decomposed, tag="foley")
    assert len(store.load("fake").paths) == 1
    assert embedder.calls == 1  # the same file under the other spelling is not embedded again
    assert store.load("fake").tags == [["foley"]]

    (composed / "one.wav").unlink()
    index_folder(store, embedder, decomposed)
    assert store.load("fake").paths == []


def test_index_can_tag_files_that_are_already_indexed(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    write_wav(lib / "one.wav", 1, seconds=1.0)
    store = Store(tmp_path / "index.db")
    embedder = FakeEmbedder()

    index_folder(store, embedder, lib)
    assert store.load("fake").tags == [[]]

    index_folder(store, embedder, lib, tag="foley")
    assert store.load("fake").tags == [["foley"]]
    assert embedder.calls == 1  # tagging does not re-embed


def test_feedback_keeps_the_latest_vote_and_survives_a_restart(tmp_path):
    from audio_embed.feedback import Feedback

    log = tmp_path / "feedback.jsonl"
    drone = {"kind": "text", "text": "a dark drone"}
    like = {"kind": "like", "path": "/lib/a.wav"}
    votes = Feedback(log)
    votes.record(drone, "/lib/a.wav", 1, model="clap", rank=1)
    votes.record(drone, "/lib/b.wav", -1, model="clap", rank=2)
    votes.record(like, "/lib/b.wav", 1)
    votes.record(drone, "/lib/b.wav", 0)  # cleared

    for reopened in (votes, Feedback(log)):
        assert reopened.verdict(drone, "/lib/a.wav") == 1
        assert reopened.verdict(drone, "/lib/b.wav") == 0
        assert reopened.verdict(like, "/lib/b.wav") == 1
        assert reopened.verdict({"kind": "text", "text": "something else"}, "/lib/a.wav") == 0
    assert len(log.read_text().splitlines()) == 4  # append-only: every change is kept


def test_feedback_ignores_a_line_cut_short(tmp_path):
    from audio_embed.feedback import Feedback

    log = tmp_path / "feedback.jsonl"
    query = {"kind": "text", "text": "rain"}
    Feedback(log).record(query, "/lib/a.wav", 1)
    with log.open("a") as f:
        f.write('{"time": "2026-10-08", "query": {"kind"')
    assert Feedback(log).verdict(query, "/lib/a.wav") == 1


def test_rank_keep_mask_limits_which_files_are_returned(store):
    matrix = store.load("m")
    only_b = np.array([False, True])
    assert [h.path for h in search.rank(matrix, unit(1, 0, 0), k=10, keep=only_b)] == ["/lib/b.wav"]
    assert search.rank(matrix, unit(1, 0, 0), k=10, keep=np.array([False, False])) == []


def test_file_means_gives_one_unit_vector_per_file(store):
    means = search.file_means(store.load("m"))
    assert means.shape == (2, 3)
    np.testing.assert_allclose(means[0], unit(1, 1, 0), atol=1e-3)
    np.testing.assert_allclose(means[1], unit(0, 0, 1), atol=1e-3)


def test_classify_assigns_each_file_its_closest_category():
    from audio_embed import classify

    files = np.stack([unit(1, 0.1, 0), unit(0, 0.2, 1)])
    categories = np.stack([unit(0, 0, 1), unit(1, 0, 0)])
    assert classify.assign(files, categories).tolist() == [1, 0]


def test_custom_category_list_replaces_the_defaults(tmp_path):
    from audio_embed import classify

    assert classify.load_categories(tmp_path / "missing.json") == classify.DEFAULT_CATEGORIES
    custom = tmp_path / "categories.json"
    custom.write_text('[{"name": "Þoka", "description": "thick fog horn"}]', encoding="utf-8")
    assert classify.load_categories(custom) == [("Þoka", "thick fog horn")]


def test_filters_match_any_value_within_a_kind_and_every_kind():
    from audio_embed import facets

    index = facets.build([
        {"sound": ["Drone"], "type": ["loop"]},
        {"sound": ["Impact"], "type": ["one-shot"]},
        {"sound": ["Drone"], "type": ["one-shot"]},
        {},
    ])
    assert facets.mask(index, 4, {"sound": {"Drone"}}).tolist() == [True, False, True, False]
    assert facets.mask(index, 4, {"sound": {"Drone", "Impact"}}).tolist() == [True, True, True, False]
    assert facets.mask(index, 4, {"sound": {"Drone"}, "type": {"one-shot"}}).tolist() == [False, False, True, False]
    assert facets.mask(index, 4, {"sound": {"Nothing"}}).tolist() == [False] * 4
    assert facets.mask(index, 4, {}).tolist() == [True] * 4


def test_filter_summary_counts_values_and_skips_free_text_kinds():
    from audio_embed import facets

    index = facets.build(
        [{"sound": ["Drone"], "note": [f"take {n}"]} for n in range(facets.MAX_DISTINCT + 1)] + [{"sound": ["Impact"]}]
    )
    assert facets.summary(index) == [
        {"kind": "sound", "values": [{"value": "Drone", "count": facets.MAX_DISTINCT + 1}, {"value": "Impact", "count": 1}]}
    ]


def kinds(pairs):
    out = {}
    for kind, value in pairs:
        out.setdefault(kind, set()).add(value)
    return out


def test_labels_from_a_name_and_its_folders():
    from pathlib import Path

    from audio_embed.labels import from_path

    found = kinds(from_path(Path("/lib/Show/Bounce/DRMs/Kick thump 82bpm _01.aif")))
    assert found == {"folder": {"DRMs"}, "bpm": {"82"}, "category": {"kick", "drums"}}

    found = kinds(from_path(Path("/lib/packs/Kore - Grooves/loop/Drums/RT_Drum_Loop_47_Sarah_135_loop.wav")))
    assert found["type"] == {"loop"} and found["bpm"] == {"135"}

    found = kinds(from_path(Path("/lib/Drum Loops/Bouncy 166.5/Bouncy Cut 1.wav")))
    assert found["type"] == {"loop"} and found["bpm"] == {"166.5"}

    found = kinds(from_path(Path("/lib/one_shot/Snare/Jazz_Kit_Snare_Rim_Click_2_one_shot.wav")))
    assert found["type"] == {"one-shot"} and "bpm" not in found and found["category"] == {"snare"}

    found = kinds(from_path(Path("/lib/Vocal Chops/SD_VoxChop_PercLoop_Proud_100_F.wav")))
    assert found["category"] == {"vocal", "percussion"} and found["type"] == {"loop"}
    assert kinds(from_path(Path("/lib/GS 2.0/808hat.aif")))["category"] == {"808", "hat"}

    # A number is only read as a tempo when the file is a loop or says "bpm".
    assert "bpm" not in kinds(from_path(Path("/lib/Hits/Impact 120.wav")))


def test_key_needs_an_accidental_a_mode_or_brackets():
    from pathlib import Path

    from audio_embed.labels import from_path

    def keys(name):
        return kinds(from_path(Path("/lib") / name)).get("key", set())

    assert keys("808 20 [C#] - Unlock It.wav") == {"C#"}
    assert keys("Deep tonal Em Bandpass.wav") == {"Em"}
    assert keys("Pad Bbmin soft.wav") == {"Bbm"}
    assert keys("Sub [F].wav") == {"F"}
    assert keys("Ambiance New A.wav") == set()  # a take name, not a key
    assert keys("violin_D#3_forte.wav") == set()  # a note of a sampled instrument


def test_category_words_survive_decomposed_accents():
    import unicodedata
    from pathlib import Path

    from audio_embed.labels import from_path

    decomposed = unicodedata.normalize("NFD", "/lib/Upptökur/Kór/Jón/Hljómar/taka.wav")
    assert kinds(from_path(Path(decomposed)))["category"] == {"choir", "chord"}


def test_labels_from_embedded_tags():
    from audio_embed.labels import from_metadata

    found = kinds(from_metadata({
        "genre": "Industry and Machines", "date": "2021-11-09", "tbpm": "134.0001",
        "encoded_by": "ZOOM Handy Recorder H2n", "comment": "Steam machine - running  with hits",
    }))
    assert found == {
        "genre": {"Industry and Machines"}, "year": {"2021"}, "bpm": {"134"},
        "encoded by": {"ZOOM Handy Recorder H2n"}, "description": {"Steam machine - running with hits"},
    }
    assert from_metadata({"tbpm": "0", "comment": "www.example.com info@example.com"}) == set()
    assert from_metadata({"comment": "Made with Sony ACID Pro 5.0"}) == set()
    assert from_metadata({"comment": "Cover (front)"}) == set()


def test_labels_from_location():
    from pathlib import Path

    from audio_embed.labels import from_location

    locations = {
        "/lib/3rd Party": {"source": "sample pack", "pack": "{1}"},
        "/lib/3rd Party/Drums/My own": {"source": "own recording"},
    }
    assert from_location(Path("/lib/3rd Party/Drums/Zero-G/Loops/a.wav"), locations) == {
        ("source", "sample pack"), ("pack", "Zero-G"),
    }
    assert from_location(Path("/lib/3rd Party/Drums/a.wav"), locations) == {("source", "sample pack")}
    assert ("source", "own recording") in from_location(Path("/lib/3rd Party/Drums/My own/a.wav"), locations)
    assert from_location(Path("/elsewhere/a.wav"), locations) == set()


def test_labels_round_trip_and_are_replaced(store):
    assert store.paths() == ["/lib/a.wav", "/lib/b.wav"]
    store.set_labels({"/lib/a.wav": {("bpm", "82"), ("category", "kick"), ("category", "drums")}})
    store.set_labels({"/lib/a.wav": {("bpm", "90")}, "/lib/not-indexed.wav": {("bpm", "1")}})
    assert store.load("m").labels == [{"bpm": ["90"]}, {}]


def test_tempos_are_grouped_into_ranges():
    from audio_embed import facets

    assert facets.tempo_range("165.5") == "160 to 179"
    assert facets.tempo_range("60") == "60 to 79"
    assert facets.tempo_range("fast") is None


def test_filter_summary_orders_kinds_and_natural_values():
    from audio_embed import facets

    index = facets.build([
        {"zeta": ["z"], "bpm": ["100 to 119"], "sound": ["Drone"], "year": ["2020"]},
        {"bpm": ["80 to 99"], "sound": ["Drone"], "year": ["2004"]},
        {"bpm": ["100 to 119"], "sound": ["Impact"], "year": ["2020"]},
    ])
    summary = facets.summary(index)
    assert [group["kind"] for group in summary] == ["sound", "bpm", "year", "zeta"]
    by_kind = {group["kind"]: [v["value"] for v in group["values"]] for group in summary}
    assert by_kind["sound"] == ["Drone", "Impact"]  # most common first
    assert by_kind["bpm"] == ["80 to 99", "100 to 119"]  # in tempo order, not by count
    assert by_kind["year"] == ["2004", "2020"]


def test_steer_moves_a_query_toward_liked_files_and_away_from_rejected_ones():
    query, liked, rejected = unit(1, 0, 0), np.stack([unit(0, 1, 0)]), np.stack([unit(0, 0, 1)])
    steered = search.steer(query, liked, rejected, weight=1.0)
    assert steered @ liked[0] > query @ liked[0]
    assert steered @ rejected[0] < query @ rejected[0]
    np.testing.assert_allclose(np.linalg.norm(steered), 1.0, atol=1e-6)
    empty = np.zeros((0, 3), dtype=np.float32)
    np.testing.assert_allclose(search.steer(query, empty, empty, weight=2.0), query, atol=1e-6)


def test_closeness_is_nothing_at_the_floor_and_full_for_the_same_search():
    query = unit(1, 0, 0)
    others = np.stack([unit(1, 0, 0), unit(4, 3, 0), unit(3, 4, 0), unit(0, 1, 0)])  # 1, 0.8, 0.6, 0 alike
    np.testing.assert_allclose(search.closeness(query, others, floor=0.6), [1.0, 0.5, 0.0, 0.0], atol=1e-6)


def test_steer_is_pulled_less_by_a_vote_that_counts_less():
    query, liked = unit(1, 0, 0), np.stack([unit(0, 1, 0), unit(0, 0, 1)])
    empty = np.zeros((0, 3), dtype=np.float32)
    steered = search.steer(query, liked, empty, weight=1.0, liked_counts=np.array([1.0, 0.25]))
    np.testing.assert_allclose(steered[1] / steered[2], 4.0, atol=1e-5)


def test_steer_pulls_only_as_hard_as_its_closest_vote_counts():
    query, liked = unit(1, 0, 0), np.stack([unit(0, 1, 0)])
    empty = np.zeros((0, 3), dtype=np.float32)
    own = search.steer(query, liked, empty, weight=1.0)
    borrowed = search.steer(query, liked, empty, weight=1.0, liked_counts=np.array([0.5]))
    np.testing.assert_allclose(own[1] / own[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(borrowed[1] / borrowed[0], 0.5, atol=1e-6)


def test_feedback_lists_the_votes_of_every_text_search(tmp_path):
    from audio_embed.feedback import Feedback

    votes = Feedback(tmp_path / "feedback.jsonl")
    votes.record({"kind": "text", "text": "drone"}, "/lib/a.wav", 1)
    votes.record({"kind": "text", "text": "drone"}, "/lib/b.wav", -1)
    votes.record({"kind": "text", "text": "rain"}, "/lib/c.wav", 1)
    votes.record({"kind": "text", "text": "wind"}, "/lib/d.wav", 1)
    votes.record({"kind": "text", "text": "wind"}, "/lib/d.wav", 0)  # cleared: no standing vote left
    votes.record({"kind": "like", "path": "/lib/a.wav"}, "/lib/e.wav", 1)
    assert votes.by_text() == {"drone": {"/lib/a.wav": 1, "/lib/b.wav": -1}, "rain": {"/lib/c.wav": 1}}


def borrowing_votes(tmp_path):
    from audio_embed.feedback import Feedback

    votes = Feedback(tmp_path / "feedback.jsonl")
    votes.record({"kind": "text", "text": "drone"}, "/lib/a.wav", 1)
    votes.record({"kind": "text", "text": "drone"}, "/lib/b.wav", -1)
    votes.record({"kind": "text", "text": "rain"}, "/lib/c.wav", 1)
    votes.record({"kind": "text", "text": "dark drone"}, "/lib/d.wav", 1)
    votes.record({"kind": "like", "path": "/lib/a.wav"}, "/lib/e.wav", 1)
    # "dark drone" is 0.8 like "drone" and 0.6 like "rain".
    return votes, {"drone": unit(1, 0, 0), "rain": unit(0, 1, 0), "dark drone": unit(4, 3, 0)}


def test_a_search_borrows_the_votes_of_a_similar_search_and_counts_them_for_less(tmp_path):
    from audio_embed.server import counted_votes

    votes, vectors = borrowing_votes(tmp_path)
    asked = {"kind": "text", "text": "dark drone"}
    counted = counted_votes(votes, asked, vectors["dark drone"], vectors.__getitem__, floor=0.6)
    assert sorted((path, verdict, round(count, 6)) for path, verdict, count in counted) == [
        ("/lib/a.wav", 1, 0.5),
        ("/lib/b.wav", -1, 0.5),
        ("/lib/d.wav", 1, 1.0),  # its own vote; nothing from "rain", which is no closer than the floor
    ]


def test_nothing_is_borrowed_without_a_floor_or_for_a_find_similar_search(tmp_path):
    from audio_embed.server import counted_votes

    votes, vectors = borrowing_votes(tmp_path)
    asked = {"kind": "text", "text": "dark drone"}
    assert counted_votes(votes, asked, vectors["dark drone"], vectors.__getitem__, floor=None) == [("/lib/d.wav", 1, 1.0)]
    like = {"kind": "like", "path": "/lib/a.wav"}
    assert counted_votes(votes, like, unit(1, 0, 0), vectors.__getitem__, floor=0.6) == [("/lib/e.wav", 1, 1.0)]


def test_feedback_lists_the_standing_votes_of_one_search(tmp_path):
    from audio_embed.feedback import Feedback

    votes = Feedback(tmp_path / "feedback.jsonl")
    drone = {"kind": "text", "text": "drone"}
    votes.record(drone, "/lib/a.wav", 1)
    votes.record(drone, "/lib/b.wav", -1)
    votes.record(drone, "/lib/c.wav", 1)
    votes.record(drone, "/lib/c.wav", 0)
    votes.record({"kind": "text", "text": "rain"}, "/lib/d.wav", 1)
    assert votes.votes(drone) == {"/lib/a.wav": 1, "/lib/b.wav": -1}


def test_collection_is_a_folder_of_shortcuts_and_never_touches_the_originals(tmp_path):
    from audio_embed.saved import Collections

    library = tmp_path / "library"
    (library / "one").mkdir(parents=True)
    (library / "two").mkdir()
    first, second = library / "one" / "hit.wav", library / "two" / "hit.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    kept = Collections(tmp_path / "collections")
    assert kept.names() == []

    kept.add("Show picks", first)
    kept.add("Show picks", first)  # adding twice changes nothing
    kept.add("Show picks", second)  # same file name, different file
    assert kept.names() == [("Show picks", 2)]
    assert sorted(kept.paths("Show picks")) == sorted([str(first), str(second)])
    assert sorted(p.name for p in kept.folder("Show picks").iterdir()) == ["hit (2).wav", "hit.wav"]
    assert (kept.folder("Show picks") / "hit (2).wav").read_bytes() == b"second"

    kept.remove("Show picks", first)
    assert kept.paths("Show picks") == [str(second)]
    assert first.read_bytes() == b"first"  # only the shortcut went away


def test_collection_remove_leaves_real_files_in_the_folder_alone(tmp_path):
    from audio_embed.saved import Collections

    target = tmp_path / "sound.wav"
    target.write_bytes(b"audio")
    kept = Collections(tmp_path / "collections")
    kept.add("Mix", target)
    stray = kept.folder("Mix") / "notes.txt"
    stray.write_text("my notes")
    kept.remove("Mix", target)
    assert stray.read_text() == "my notes"
    assert kept.paths("Mix") == []


def test_collection_names_must_be_safe_folder_names():
    from audio_embed.saved import clean_name

    assert clean_name("  Show   picks ") == "Show picks"
    for bad in ("", "   ", ".hidden", "a/b", "../up", "a:b", "x" * 61):
        assert clean_name(bad) is None


def test_moods_start_from_defaults_and_can_be_changed(tmp_path):
    from audio_embed.saved import DEFAULT_MOODS, Moods

    moods = Moods(tmp_path / "moods.json")
    assert moods.all() == DEFAULT_MOODS
    moods.save("þoka", "thick fog, distant horn")
    moods.save("dark", "a very dark drone")  # replaces the default of that name
    names = {m["name"]: m["description"] for m in Moods(tmp_path / "moods.json").all()}
    assert names["þoka"] == "thick fog, distant horn"
    assert names["dark"] == "a very dark drone"
    assert len(names) == len(DEFAULT_MOODS) + 1
    moods.remove("dark")
    assert "dark" not in {m["name"] for m in moods.all()}


def test_duplicates_are_files_with_identical_contents(tmp_path):
    from audio_embed import duplicates

    files = {"a.wav": b"same sound", "b.wav": b"same sound", "c.wav": b"SAME SOUND", "d.wav": b"longer, different",
             "e.wav": b"", "f.wav": b""}
    paths = []
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
        paths.append(str(tmp_path / name))
    paths.append(str(tmp_path / "gone.wav"))  # indexed once, deleted since

    known = {}
    assert duplicates.groups(paths, known) == [([0, 1], 10)]  # c is the same size but different; empty files are skipped
    assert len(known) == 3  # only the three files that share a size were read
    assert duplicates.groups(paths, known) == [([0, 1], 10)]


def test_outline_is_one_byte_per_bucket_scaled_to_the_loudest_point():
    quiet_then_loud = np.concatenate([np.full(500, 0.1, dtype=np.float32), np.full(500, -0.4, dtype=np.float32)])
    levels = audio.outline(quiet_then_loud, buckets=10)
    assert levels.dtype == np.uint8
    assert levels.tolist() == [64] * 5 + [255] * 5
    assert audio.outline(np.zeros(100, dtype=np.float32), buckets=10).tolist() == [0] * 10
    assert len(audio.outline(np.ones(3, dtype=np.float32), buckets=10)) == 3  # never more buckets than samples
    assert len(audio.outline(np.zeros(0, dtype=np.float32))) == 0


def test_outlines_are_stored_per_file_version(tmp_path):
    from audio_embed.outlines import Outlines

    outlines = Outlines(tmp_path / "outlines.db")
    outlines.put("/lib/a.wav", 100, 1.5, np.array([0, 128, 255], dtype=np.uint8))
    assert outlines.get("/lib/a.wav", 100, 1.5) == [0.0, 0.502, 1.0]
    assert outlines.get("/lib/a.wav", 101, 1.5) is None  # the file changed since
    assert outlines.get("/lib/b.wav", 100, 1.5) is None
    outlines.put("/lib/a.wav", 101, 2.5, np.array([255], dtype=np.uint8))
    assert Outlines(tmp_path / "outlines.db").stamps() == {"/lib/a.wav": (101, 2.5)}


def test_indexing_stores_an_outline_and_the_outlines_command_fills_in_the_rest(tmp_path, capsys):
    from argparse import Namespace

    from audio_embed.cli import cmd_outlines
    from audio_embed.outlines import Outlines, beside

    lib = tmp_path / "lib"
    lib.mkdir()
    write_wav(lib / "one.wav", 1, seconds=1.5)
    write_wav(lib / "two.wav", 2, seconds=0.5)
    db = tmp_path / "data" / "index.db"
    store = Store(db)
    index_folder(store, FakeEmbedder(), lib)

    outlines = Outlines(beside(db))
    assert sorted(outlines.stamps()) == [str(lib / "one.wav"), str(lib / "two.wav")]
    stat = (lib / "one.wav").stat()
    levels = outlines.get(str(lib / "one.wav"), stat.st_size, stat.st_mtime)
    assert len(levels) == 800 and max(levels) == 1.0

    # Lose one outline and change the other file: the command rebuilds exactly those two.
    outlines.db.execute("DELETE FROM outlines WHERE path = ?", (str(lib / "one.wav"),))
    outlines.db.commit()
    write_wav(lib / "two.wav", 2, seconds=0.75)
    cmd_outlines(Namespace(db=db, jobs=2), store)
    assert "1 outlines stored, 2 to build" in capsys.readouterr().out
    fresh = Outlines(beside(db))
    for name in ("one.wav", "two.wav"):
        stat = (lib / name).stat()
        assert fresh.get(str(lib / name), stat.st_size, stat.st_mtime) is not None
    cmd_outlines(Namespace(db=db, jobs=2), store)
    assert "2 outlines stored, 0 to build" in capsys.readouterr().out


def fake_transcode(calls):
    def transcode(source, out):
        calls.append(source.name)
        out.write_bytes(b"aac:" + source.read_bytes()[:4])  # 8 bytes: smaller than any original longer than that
    return transcode


def render_drive(tmp_path, calls=None):
    """A render folder on a stand-in drive, with a stand-in transcoder that writes 8 bytes."""
    from audio_embed import renders

    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    renders.setup(drive / "renders")
    return renders.Renders(tmp_path / "data" / "renders.db", drive / "renders", fake_transcode([] if calls is None else calls))


def test_render_folder_is_only_written_to_when_its_marker_is_there(tmp_path):
    from audio_embed import renders

    with pytest.raises(RuntimeError, match="drive connected"):
        renders.setup(tmp_path / "unplugged" / "renders")  # the drive it should sit on is not there
    assert not (tmp_path / "unplugged").exists()

    source = tmp_path / "a.wav"
    source.write_bytes(b"sound")
    stat = source.stat()
    # A folder that merely exists is not a render folder: an unplugged drive must not be recreated on this disk.
    (tmp_path / "plain").mkdir()
    for root in (tmp_path / "plain", None):
        library = renders.Renders(tmp_path / "data" / "renders.db", root, fake_transcode([]))
        assert not library.connected()
        with pytest.raises(RuntimeError):
            library.ensure(source, stat.st_size, stat.st_mtime)
    assert list((tmp_path / "plain").iterdir()) == []

    renders.save_root(tmp_path / "data" / "settings.json", tmp_path / "plain")
    assert renders.saved_root(tmp_path / "data" / "settings.json") == tmp_path / "plain"
    assert renders.saved_root(tmp_path / "data" / "missing.json") is None


def test_identical_files_share_a_render_and_a_changed_file_gets_a_new_one(tmp_path):
    calls = []
    library = render_drive(tmp_path, calls)
    lib = tmp_path / "lib"
    lib.mkdir()
    for name, content in (("a.wav", b"the same sound"), ("copy of a.wav", b"the same sound"), ("b.wav", b"another sound")):
        (lib / name).write_bytes(content)

    def render(name):
        stat = (lib / name).stat()
        return library.ensure(lib / name, stat.st_size, stat.st_mtime)

    assert [render(n) for n in ("a.wav", "copy of a.wav", "b.wav")] == ["made", "shared", "made"]
    assert calls == ["a.wav", "b.wav"]
    assert library.copy_of(str(lib / "a.wav")) == library.copy_of(str(lib / "copy of a.wav"))
    assert library.copy_of(str(lib / "a.wav")).read_bytes() == b"aac:the "

    old = library.copy_of(str(lib / "a.wav"))
    (lib / "a.wav").write_bytes(b"a new bounce")
    assert render("a.wav") == "made"
    assert library.copy_of(str(lib / "a.wav")).read_bytes() == b"aac:a ne"
    assert old.exists()  # still the render of "copy of a.wav"; nothing is ever deleted from the drive
    assert not list(library.root.rglob("*.partial"))


def test_render_plays_once_the_original_is_online_only_and_not_without_the_drive(tmp_path):
    from audio_embed import renders
    from audio_embed.server import Library

    library = render_drive(tmp_path)
    renders.save_root(tmp_path / "data" / "settings.json", library.root)
    source = tmp_path / "a.wav"
    source.write_bytes(b"a long enough sound")
    stat = source.stat()
    library.ensure(source, stat.st_size, stat.st_mtime)
    library.back_up()

    served = Library(tmp_path / "data" / "index.db")
    assert served.heard_from(source) == ("original", source)

    source.write_bytes(b"")  # what a file looks like after Dropbox makes it online-only
    assert not audio.is_local(source)
    assert served.heard_from(source) == ("render", library.copy_of(str(source)))
    assert served.heard_from(tmp_path / "never rendered.wav") == ("none", None)
    # Only a render made from the asked-for version counts when a version is given.
    assert library.copy_of(str(source), (stat.st_size, stat.st_mtime)) is not None
    assert library.copy_of(str(source), (stat.st_size + 1, stat.st_mtime)) is None

    library.root.rename(tmp_path / "drive" / "elsewhere")  # the drive is unplugged
    assert served.heard_from(source) == ("none", None)

    # The catalogue's copy on the drive is enough to match renders to files again.
    restored = renders.Renders(tmp_path / "drive" / "elsewhere" / renders.BACKUP, tmp_path / "drive" / "elsewhere")
    assert restored.copy_of(str(source)) is not None


def test_render_files_makes_what_is_missing_and_drops_stale_rows_without_the_drive(tmp_path, capsys):
    from audio_embed.cli import render_files

    calls = []
    library = render_drive(tmp_path, calls)
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a.wav").write_bytes(b"one one one")
    (lib / "b.wav").write_bytes(b"two two two")
    (lib / "cloud.wav").write_bytes(b"")
    paths = [str(lib / n) for n in ("a.wav", "b.wav", "cloud.wav", "deleted.wav")]

    render_files(library, paths, jobs=2)
    out = capsys.readouterr().out
    assert "2 made" in out and "1 files are online-only and have no render yet" in out
    render_files(library, paths, jobs=2)
    assert sorted(calls) == ["a.wav", "b.wav"]  # nothing is converted twice

    # b goes online-only: it keeps its render. a changes while the drive is away: its row goes.
    (lib / "b.wav").write_bytes(b"")
    (lib / "a.wav").write_bytes(b"one, bounced again")
    library.root.rename(tmp_path / "drive" / "away")
    render_files(library, paths)
    assert "1 files are waiting for a render" in capsys.readouterr().out
    assert sorted(library.stamps()) == [str(lib / "b.wav")]
    (tmp_path / "drive" / "away").rename(library.root)
    assert library.copy_of(str(lib / "b.wav")) is not None and library.copy_of(str(lib / "a.wav")) is None


def test_a_real_render_is_a_playable_stereo_file(tmp_path):
    from audio_embed import renders

    drive = tmp_path / "drive"
    drive.mkdir()
    renders.setup(drive / "renders")
    library = renders.Renders(tmp_path / "data" / "renders.db", drive / "renders")
    source = tmp_path / "wide.wav"
    write_wav(source, 6, seconds=0.5)
    stat = source.stat()
    assert library.ensure(source, stat.st_size, stat.st_mtime) == "made"
    assert audio.channel_count(library.copy_of(str(source))) == 2


def test_index_leaves_online_only_files_alone(tmp_path, capsys):
    lib = tmp_path / "lib"
    lib.mkdir()
    write_wav(lib / "one.wav", 1, seconds=1.0)
    write_wav(lib / "two.wav", 1, seconds=1.0)
    store = Store(tmp_path / "index.db")
    embedder = FakeEmbedder()
    index_folder(store, embedder, lib)
    capsys.readouterr()

    (lib / "two.wav").write_bytes(b"")  # made online-only after it was indexed
    (lib / "never here.wav").write_bytes(b"")
    index_folder(store, embedder, lib)
    printed = capsys.readouterr()
    assert "1 already indexed" in printed.out and "0 to embed, 2 online-only" in printed.out
    assert "FAILED" not in printed.err
    assert store.load("fake").paths == [str(lib / "one.wav"), str(lib / "two.wav")]  # still searchable
    assert embedder.calls == 2


def test_labels_from_tags_are_kept_while_a_file_is_online_only(tmp_path):
    from audio_embed import labels

    lib = tmp_path / "Foley" / "Doors"
    lib.mkdir(parents=True)
    path = lib / "creak 90bpm.wav"
    path.write_bytes(b"")
    earlier = {("artist", "Some Library"), ("year", "2014"), ("folder", "Old name"), ("source", "foley")}
    locations = {str(tmp_path / "Foley"): {"source": "downloaded sfx"}}
    found = labels.read(path, locations, earlier)
    assert {("artist", "Some Library"), ("year", "2014")} <= found
    assert ("source", "downloaded sfx") in found and ("source", "foley") not in found
    assert ("folder", "Doors") in found and ("folder", "Old name") not in found
    assert ("category", "door") in found and ("bpm", "90") in found


def test_waveform_of_an_online_only_file_is_the_one_stored_while_it_was_here(tmp_path):
    from audio_embed.outlines import Outlines, beside
    from audio_embed.server import Library

    source = tmp_path / "a.wav"
    write_wav(source, 1, seconds=0.5)
    served = Library(tmp_path / "data" / "index.db")
    drawn = served.peaks(source)
    assert len(drawn) == 800

    source.write_bytes(b"")
    assert served.peaks(source) == drawn
    assert served.peaks(source, build=False) == drawn
    never = tmp_path / "never.wav"
    never.write_bytes(b"")
    assert served.peaks(never) is None
    assert Outlines(beside(tmp_path / "data" / "index.db")).latest(str(never)) is None


def test_duplicates_command_writes_every_set_to_a_spreadsheet(tmp_path, capsys):
    import csv
    from argparse import Namespace

    from audio_embed.cli import cmd_duplicates

    lib = tmp_path / "lib"
    (lib / "Kór").mkdir(parents=True)
    for name, content in (("a.wav", b"x" * 3000), ("Kór/á copy.wav", b"x" * 3000), ("b.wav", b"y" * 3000), ("big 1.wav", b"z" * 9000), ("big 2.wav", b"z" * 9000)):
        (lib / name).write_bytes(content)
    db = tmp_path / "data" / "index.db"
    store = Store(db)
    for path in sorted(lib.rglob("*.wav")):
        store.put("fake", str(path), 1, 1.0, 1.0, [(0.0, 1.0)], np.array([unit(1, 0, 0)]))

    cmd_duplicates(Namespace(db=db, out=None), store)
    assert "2 sets of identical files, 2 spare copies" in capsys.readouterr().out
    with (tmp_path / "data" / "duplicates.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["set"], r["copies"], r["name"]) for r in rows] == [
        ("1", "2", "big 1.wav"), ("1", "2", "big 2.wav"), ("2", "2", "á copy.wav"), ("2", "2", "a.wav"),
    ]
    assert rows[0]["spare_mb"] == "0.01" and rows[2]["folder"] == str(lib / "Kór")


def test_preview_of_a_one_channel_file_that_is_not_labelled_mono(tmp_path):
    import subprocess

    # Some tools write a single channel as "front left", which the AAC encoder refuses as it stands.
    source, out = tmp_path / "left.wav", tmp_path / "left.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5:sample_rate=32000",
         "-af", "pan=FL|c0=c0", "-c:a", "pcm_f32le", str(source)],
        check=True,
    )
    audio.write_preview(source, out)
    assert audio.channel_count(out) == 1
    before, after = (np.abs(audio.decode(p, 8000)).max() for p in (source, out))
    assert after == pytest.approx(before, rel=0.1)  # not silenced or attenuated on the way


def test_a_render_is_never_larger_than_its_original(tmp_path):
    from audio_embed import duplicates

    library = render_drive(tmp_path)
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in ("hit.wav", "same hit.wav"):
        (lib / name).write_bytes(b"tiny")  # the stand-in AAC copy is 8 bytes: larger than this

    def render(name):
        stat = (lib / name).stat()
        return library.ensure(lib / name, stat.st_size, stat.st_mtime)

    assert [render("hit.wav"), render("same hit.wav")] == ["made", "shared"]
    copy = library.copy_of(str(lib / "hit.wav"))
    assert copy.suffix == ".wav" and copy.read_bytes() == b"tiny"  # an exact copy, under the original's own suffix
    assert library.copy_of(str(lib / "same hit.wav")) == copy
    assert [p.name for p in library.root.rglob("*") if p.is_file() and not p.name.startswith(".")] == [copy.name]

    # Renders made before this rule: an AAC copy larger than its original is replaced by the original's bytes.
    files = {"old.wav": b"short", "old.m4a": b"small", "away.wav": b"gone!"}
    for name, content in files.items():
        (lib / name).write_bytes(content)
        stat = (lib / name).stat()
        digest = duplicates.digest(str(lib / name))
        library.file(digest).parent.mkdir(exist_ok=True)
        library.file(digest).write_bytes(b"an oversized aac copy")
        library.db.execute("INSERT INTO renders (path, size, mtime, digest) VALUES (?, ?, ?, ?)", (str(lib / name), stat.st_size, stat.st_mtime, digest))
    library.db.commit()
    (lib / "away.wav").write_bytes(b"")  # online-only: there is nothing to copy from, so its render stays
    assert library.shrink() == (2, 1)
    assert library.copy_of(str(lib / "old.wav")).name.endswith(".wav")
    assert library.copy_of(str(lib / "old.wav")).read_bytes() == b"short"
    assert library.copy_of(str(lib / "old.m4a")).read_bytes() == b"small"
    assert library.copy_of(str(lib / "away.wav")).read_bytes() == b"an oversized aac copy"
    assert len([p for p in library.root.rglob("*") if p.is_file() and not p.name.startswith(".")]) == 4
    assert library.shrink() == (0, 1)


def test_a_real_very_short_sample_is_kept_as_it_is(tmp_path):
    from audio_embed import renders

    drive = tmp_path / "drive"
    drive.mkdir()
    renders.setup(drive / "renders")
    library = renders.Renders(tmp_path / "data" / "renders.db", drive / "renders")
    source = tmp_path / "click.wav"
    write_wav(source, 1, seconds=0.01)
    stat = source.stat()
    library.ensure(source, stat.st_size, stat.st_mtime)
    copy = library.copy_of(str(source))
    assert copy.suffix == ".wav" and copy.read_bytes() == source.read_bytes()


def test_quality_is_read_from_the_codec_not_the_extension():
    from audio_embed import labels

    for codec, suffix, want in [
        ("pcm_s24le", ".wav", "lossless"),
        ("pcm_f32be", ".aif", "lossless"),
        ("flac", ".flac", "lossless"),
        ("alac", ".m4a", "lossless"),  # Apple Lossless shares its extension with AAC
        ("aac", ".m4a", "lossy"),
        ("mp3", ".mp3", "lossy"),
        ("vorbis", ".ogg", "lossy"),
        ("adpcm_ms", ".wav", "lossy"),  # a WAV is only a wrapper, and this one holds compressed audio
    ]:
        assert labels.quality(codec, suffix) == want, (codec, suffix)


def test_quality_falls_back_to_the_extension_when_the_codec_is_unknown():
    from audio_embed import labels

    assert labels.quality("", ".wav") == "lossless"
    assert labels.quality("", ".AIFF") == "lossless"
    assert labels.quality("", ".MP3") == "lossy"
    assert labels.quality("", ".m4a") == "lossy"


def test_a_file_is_labelled_with_the_quality_of_what_it_holds(tmp_path):
    import subprocess

    from audio_embed import labels

    plain = tmp_path / "plain.wav"
    write_wav(plain, 1, seconds=0.5)
    squeezed = tmp_path / "squeezed.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(plain), "-c:a", "adpcm_ms", str(squeezed)], check=True)
    assert {v for k, v in labels.read(plain, {}) if k == "quality"} == {"lossless"}
    assert {v for k, v in labels.read(squeezed, {}) if k == "quality"} == {"lossy"}


def test_quality_of_an_online_only_file_is_the_one_stored_while_it_was_here(tmp_path):
    from audio_embed import labels

    squeezed = tmp_path / "squeezed.wav"
    squeezed.write_bytes(b"")
    found = labels.read(squeezed, {}, {("quality", "lossy")})
    assert {v for k, v in found if k == "quality"} == {"lossy"}  # not what its extension suggests

    never_read = tmp_path / "never read.mp3"
    never_read.write_bytes(b"")
    assert ("quality", "lossy") in labels.read(never_read, {}, set())


# ---------- hiding files from searches


def test_a_hidden_file_is_left_out_and_stays_hidden(tmp_path):
    from audio_embed.hidden import Hidden

    paths = ["/lib/a.wav", "/lib/b.wav"]
    hidden = Hidden(tmp_path / "hidden.json")
    assert not hidden.mask(paths).any()
    hidden.hide("/lib/b.wav")
    assert hidden.mask(paths).tolist() == [False, True]
    assert Hidden(tmp_path / "hidden.json").mask(paths).tolist() == [False, True]
    hidden.show("/lib/b.wav")
    assert not Hidden(tmp_path / "hidden.json").mask(paths).any()


def test_a_hidden_folder_hides_everything_under_it_and_nothing_beside_it(tmp_path):
    from audio_embed.hidden import Hidden

    paths = ["/lib/Lög/a.wav", "/lib/Lög/deeper/b.wav", "/lib/Lög 2/c.wav", "/lib/d.wav"]
    hidden = Hidden(tmp_path / "hidden.json")
    hidden.hide_folder("/lib/Lög")
    assert hidden.mask(paths).tolist() == [True, True, False, False]
    assert hidden.folder_of("/lib/Lög/deeper/b.wav") == "/lib/Lög"
    assert hidden.folder_of("/lib/d.wav") is None
    hidden.show_folder("/lib/Lög")
    assert not hidden.mask(paths).any()


def test_hidden_paths_match_however_their_accents_are_written(tmp_path):
    import unicodedata

    from audio_embed.hidden import Hidden

    hidden = Hidden(tmp_path / "hidden.json")
    hidden.hide_folder(unicodedata.normalize("NFD", "/lib/Lög"))
    hidden.hide(unicodedata.normalize("NFC", "/lib/Píanó.wav"))
    paths = [unicodedata.normalize("NFC", "/lib/Lög/a.wav"), unicodedata.normalize("NFD", "/lib/Píanó.wav")]
    assert hidden.mask(paths).all()


@pytest.mark.parametrize("path", [
    "/show/audio/Avril Lavigne - Sk8er Boi (Official Audio).mp3",
    "/show/audio/Fleetwood mac - Dreams.mp3",
    "/show/Downloads/01 - Elly Vilhjálms - Hvít jól.flac",
    "/show/Bounce/Lög/MP3 to WAW/Just FriendsM2W.wav",
    "/show/Bounce/Lög/Køp bananer 152.wav",
    "/show/audio/Rihanna - Umbrella (Audio) ft. Jay-Z.mp3",
])
def test_names_that_look_like_songs(path):
    from audio_embed.hidden import looks_like_song

    assert looks_like_song(path, 200.0)


@pytest.mark.parametrize("path", [
    "/show/Bounce/Textures/Kór 1 - NV FOLLOW.wav",           # the user's own bounce: a dash, but not compressed
    "/show/Textures/2 Sello stuff Skrap - ENV FOLLOW.wav",
    "/lib/Foley/Ambiance/AMB_Nature_Forest_General_Trail_Humidity_Fog.wav",
    "/lib/Foley/Ambiance/interior-city-apartment-53658.mp3",
    "/lib/Úr upptökum/IPHONE/New Recording 5.mp3",
    "/lib/packs/loops/SS_SL_138_drum_loop_bananamania.wav",
])
def test_names_that_do_not_look_like_songs(path):
    from audio_embed.hidden import looks_like_song

    assert not looks_like_song(path, 200.0)


def test_a_short_file_is_never_taken_for_a_song():
    from audio_embed.hidden import looks_like_song

    assert not looks_like_song("/packs/Cutty Ranks - The Stopper.mp3", 8.0)


def test_songs_are_suggested_until_hidden_or_cleared(tmp_path):
    from audio_embed.hidden import Hidden

    paths = ["/a/Fleetwood mac - Dreams.mp3", "/a/Robyn - Dancing On My Own.mp3", "/a/MRI - Sounds of a scan.mp3", "/a/drone.wav"]
    durations = [250.0, 280.0, 500.0, 90.0]
    hidden = Hidden(tmp_path / "hidden.json")
    assert hidden.suggested(paths, durations) == [0, 1, 2]
    hidden.hide(paths[0])
    hidden.not_a_song(paths[2])
    assert hidden.suggested(paths, durations) == [1]
    # Showing a hidden file that looks like a song settles it: it is not suggested again.
    hidden.show(paths[0])
    assert Hidden(tmp_path / "hidden.json").suggested(paths, durations) == [1]
