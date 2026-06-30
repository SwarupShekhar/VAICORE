#!/usr/bin/env python3
"""Unit tests for vad_segmenter.

map_words_to_regions is pure logic and fully tested here. detect_speech_regions
needs the silero-vad model (network/download) so its test is an integration test
skipped when silero_vad is not importable.
"""

import importlib.util
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vad_segmenter import VadConfig, detect_speech_regions, map_words_to_regions

_HAS_SILERO = importlib.util.find_spec("silero_vad") is not None


def _word(text, start, end, prob=0.9):
    return {"text": text, "start": start, "end": end, "prob": prob}


# ── map_words_to_regions ────────────────────────────────────────────────────────

def test_word_inside_region_is_kept():
    words = [_word("hello", 1.0, 1.5)]
    regions = [(0.5, 2.0)]
    segs = map_words_to_regions(words, regions)
    assert len(segs) == 1
    assert segs[0]["text"] == "hello"
    assert segs[0]["start"] == 0.5
    assert segs[0]["end"] == 2.0


def test_word_in_silence_gap_is_dropped():
    # midpoint 5.0 falls in the gap between the two regions → dropped.
    words = [_word("silence", 4.8, 5.2)]
    regions = [(0.0, 3.0), (7.0, 9.0)]
    segs = map_words_to_regions(words, regions)
    assert segs == []


def test_multiple_words_joined_in_time_order():
    # provided out of order; must be sorted by start and single-space joined.
    words = [
        _word("world", 1.6, 2.0),
        _word("hello", 1.0, 1.5),
    ]
    regions = [(0.5, 2.5)]
    segs = map_words_to_regions(words, regions)
    assert len(segs) == 1
    assert segs[0]["text"] == "hello world"


def test_boundary_word_assigned_by_midpoint():
    # word spans [2.5, 3.5], midpoint 3.0; region A ends at 3.0, B starts at 3.0.
    # midpoint 3.0 is contained by A (inclusive) → assigned to first region.
    words = [_word("edge", 2.5, 3.5)]
    regions = [(0.0, 3.0), (3.0, 6.0)]
    segs = map_words_to_regions(words, regions)
    assert len(segs) == 1
    assert segs[0]["start"] == 0.0  # region A

    # word with midpoint clearly inside B only goes to B.
    words2 = [_word("inb", 4.0, 5.0)]
    segs2 = map_words_to_regions(words2, regions)
    assert len(segs2) == 1
    assert segs2[0]["start"] == 3.0  # region B


def test_avg_prob_is_mean_of_member_probs():
    words = [
        _word("a", 1.0, 1.2, prob=0.4),
        _word("b", 1.3, 1.5, prob=0.6),
        _word("c", 1.6, 1.8, prob=0.8),
    ]
    regions = [(0.5, 2.0)]
    segs = map_words_to_regions(words, regions)
    assert segs[0]["avg_prob"] == pytest.approx((0.4 + 0.6 + 0.8) / 3)


def test_region_with_no_words_is_dropped():
    words = [_word("hi", 1.0, 1.4)]
    regions = [(0.5, 2.0), (5.0, 7.0)]  # second region gets nothing
    segs = map_words_to_regions(words, regions)
    assert len(segs) == 1
    assert segs[0]["start"] == 0.5


def test_region_order_preserved():
    words = [
        _word("first", 1.0, 1.4),
        _word("second", 6.0, 6.4),
    ]
    regions = [(0.5, 2.0), (5.0, 7.0)]
    segs = map_words_to_regions(words, regions)
    assert [s["text"] for s in segs] == ["first", "second"]
    assert [s["start"] for s in segs] == [0.5, 5.0]


def test_empty_words_returns_empty():
    assert map_words_to_regions([], [(0.0, 1.0)]) == []


def test_empty_regions_returns_empty():
    assert map_words_to_regions([_word("x", 0.1, 0.2)], []) == []


def test_empty_inputs_returns_empty():
    assert map_words_to_regions([], []) == []


# ── detect_speech_regions (integration) ─────────────────────────────────────────

@pytest.mark.skipif(
    not _HAS_SILERO,
    reason="integration test: requires the silero-vad package (model download)",
)
def test_detect_speech_regions_returns_tuple_list(tmp_path):
    """Tolerant smoke test: 1s noise-burst, 3s silence, 1s noise-burst @16kHz.
    Asserts the call returns a list of (float, float) tuples without error;
    does not assert region count — Silero may not flag synthetic noise as speech."""
    sr = 16000
    rng = np.random.default_rng(0)
    burst = (rng.standard_normal(sr) * 0.3).astype(np.float32)
    silence = np.zeros(3 * sr, dtype=np.float32)
    audio = np.concatenate([burst, silence, burst])
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)

    wav_path = str(tmp_path / "synthetic.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())

    regions = detect_speech_regions(wav_path, VadConfig())
    assert isinstance(regions, list)
    for r in regions:
        assert isinstance(r, tuple) and len(r) == 2
        start, end = r
        assert isinstance(start, float) and isinstance(end, float)
        assert start <= end
