import math
import wave

import numpy as np
import pytest

from audio_embed import splice


def clip(n, value):
    return np.full((n, 2), value, dtype="<i2")


def test_a_wav_is_read_back_as_it_was_written(tmp_path):
    samples = (np.arange(2000, dtype="<i2").reshape(-1, 2) - 500).copy()
    splice.write(tmp_path / "a.wav", samples)
    assert np.array_equal(splice.read(tmp_path / "a.wav"), samples)
    with wave.open(str(tmp_path / "a.wav")) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (44100, 2, 2)


def test_a_wav_of_another_shape_is_refused(tmp_path):
    with wave.open(str(tmp_path / "mono.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(b"\0\0" * 10)
    with pytest.raises(ValueError, match="44.1 kHz 16-bit stereo"):
        splice.read(tmp_path / "mono.wav")


def test_snap_moves_a_time_to_the_nearest_step():
    assert splice.snap(0.0) == 0
    assert splice.snap(0.1) == 4096        # 1.08 steps
    assert splice.snap(1.0) == 11 * 4096   # 10.77 steps
    assert splice.snap(0.04) == 0          # 0.43 steps


def test_clock_is_read_by_stable_audio_3_as_the_same_step_and_length():
    assert splice.clock(0) == "0.000000"
    assert splice.clock(45056) == "1.021677"
    for steps in (1, 2, 21, 441, 646, 882, 1292):   # up to two minutes; 441 steps is 40.96 s exactly
        n = steps * splice.STEP
        seconds = float(splice.clock(n))
        assert math.ceil(seconds * splice.SR / splice.STEP) == steps   # how many steps it makes
        assert round(seconds * splice.SR / splice.STEP) == steps       # where a part's edge lands
        assert int(round(seconds * splice.SR)) == n                    # how long its file is


def test_whole_steps_drops_what_does_not_fill_a_step():
    assert len(splice.whole_steps(clip(4096 * 3 + 100, 1))) == 4096 * 3
    assert len(splice.whole_steps(clip(4096 * 3, 1))) == 4096 * 3


def test_a_turn_puts_a_sample_first_and_turning_back_undoes_it():
    samples = np.arange(20, dtype="<i2").reshape(-1, 2).copy()
    turned = splice.turn(samples, 3)
    assert turned[0].tolist() == samples[3].tolist()
    assert turned[-1].tolist() == samples[2].tolist()
    assert np.array_equal(splice.turn_back(turned, 3), samples)


def test_only_the_part_and_its_fades_differ_from_the_original():
    original, made = clip(40000, 100), clip(40000, 9000)
    out = splice.join(original, made, 8192, 16384)
    assert out.dtype == np.dtype("<i2") and len(out) == 40000
    assert np.array_equal(out[: 8192 - splice.FADE], original[: 8192 - splice.FADE])
    assert np.array_equal(out[16384 + splice.FADE:], original[16384 + splice.FADE:])
    assert np.array_equal(out[8192:16384], made[8192:16384])
    before = out[8192 - splice.FADE: 8192, 0]
    assert 100 < before[0] < before[-1] < 9000 and np.all(np.diff(before) >= 0)
    after = out[16384: 16384 + splice.FADE, 0]
    assert 9000 > after[0] > after[-1] > 100 and np.all(np.diff(after) <= 0)


def test_a_part_at_either_end_has_no_fade_there():
    original, made = clip(20000, 100), clip(20000, 9000)
    assert np.array_equal(splice.join(original, made, 0, 4096)[:4096], made[:4096])
    assert np.array_equal(splice.join(original, made, 16000, 20000)[16000:], made[16000:])


def test_a_part_past_the_end_makes_the_clip_longer():
    original, made = clip(8192, 100), clip(20480, 9000)
    out = splice.join(original, made, 8192, 20480)
    assert len(out) == 20480
    assert np.array_equal(out[: 8192 - splice.FADE], original[: 8192 - splice.FADE])
    assert np.array_equal(out[8192:], made[8192:])


def test_a_made_clip_that_is_too_short_is_refused():
    with pytest.raises(ValueError, match="shorter clip"):
        splice.join(clip(20000, 1), clip(10000, 2), 8192, 16384)
    with pytest.raises(ValueError, match="no length"):
        splice.join(clip(20000, 1), clip(20000, 2), 8192, 8192)


def test_a_fade_out_takes_the_end_of_a_clip_down_to_nothing():
    out = splice.fade_out(clip(20000, 9000))
    assert out.dtype == np.dtype("<i2") and len(out) == 20000
    assert np.all(out[: 20000 - splice.FADE] == 9000)
    tail = out[20000 - splice.FADE:, 0]
    assert tail[0] > 8900 and tail[-1] < 100 and np.all(np.diff(tail) <= 0)
    assert len(splice.fade_out(clip(100, 9000))) == 100   # a clip shorter than the fade
