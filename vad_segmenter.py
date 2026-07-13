#!/usr/bin/env python3
"""
vad_segmenter.py — VAD-first segmentation for Bajaj Finance / Navana.ai (CLIENT002)

Two pure, independently testable functions (design §3.1):

    detect_speech_regions(channel_wav_path, cfg) -> List[(start_s, end_s)]
        Run Silero VAD on a mono 16kHz channel WAV; return ordered,
        non-overlapping speech windows in seconds.

    map_words_to_regions(words, regions) -> List[Segment]
        Assign each word to the region containing its midpoint; drop words
        in silence; emit one {text, start, end, avg_prob} segment per region.

Silence has no VAD window, so words mapped by midpoint can never land in
silence — segments are silence-free by construction.

No CLI, no I/O beyond Silero's read_audio on the channel WAV.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_SAMPLE_RATE = 16000  # channels are produced as 16kHz mono WAV (processor.py:401)


@dataclass
class VadConfig:
    """Silero VAD tunables. threshold and min_silence_ms are the live knobs."""
    threshold: float = 0.5
    min_silence_ms: int = 100    # Extract tight segments first to prevent cross-speaker merging
    min_speech_ms: int = 250
    speech_pad_ms: int = 0       # Padding is applied globally after merging to avoid overlaps


# ── Silero model — lazy module-level singleton ──────────────────────────────────

_MODEL = None


def _get_model():
    """Load the Silero VAD model once and cache it for repeated calls."""
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad
        _MODEL = load_silero_vad()
    return _MODEL


# ── VAD ─────────────────────────────────────────────────────────────────────────

def detect_speech_regions(
    channel_wav_path: str, cfg: VadConfig = VadConfig()
) -> List[Tuple[float, float]]:
    """Run Silero VAD on a mono channel WAV; return ordered, non-overlapping
    (start_s, end_s) speech windows, rounded to 3 decimals."""
    from silero_vad import read_audio, get_speech_timestamps

    model = _get_model()
    wav = read_audio(channel_wav_path, sampling_rate=_SAMPLE_RATE)
    ts = get_speech_timestamps(
        wav,
        model,
        sampling_rate=_SAMPLE_RATE,
        return_seconds=True,
        threshold=cfg.threshold,
        min_silence_duration_ms=cfg.min_silence_ms,
        min_speech_duration_ms=cfg.min_speech_ms,
        speech_pad_ms=cfg.speech_pad_ms,
    )
    regions = [(round(float(t["start"]), 3), round(float(t["end"]), 3)) for t in ts]
    regions.sort(key=lambda r: r[0])
    return regions


# ── Word → region mapping ───────────────────────────────────────────────────────

def _region_of(midpoint: float, regions: List[Tuple[float, float]]) -> Optional[int]:
    """Index of the region whose [start, end] contains midpoint, else None."""
    for i, (start, end) in enumerate(regions):
        if start <= midpoint <= end:
            return i
    return None


def map_words_to_regions(
    words: List[Dict], regions: List[Tuple[float, float]]
) -> List[Dict]:
    """Map word-level results into VAD regions by word midpoint.

    words items: {"text": str, "start": float, "end": float, "prob": float}.
    A word whose midpoint ((start+end)/2) falls in no region is dropped
    (silence / hallucination). For each region with ≥1 word, emit
    {text, start, end, avg_prob}; empty regions are dropped. Region order is
    preserved and words within a region are joined in start-time order.
    Additionally, any gap between words > 0.5s will split the segment.
    """
    buckets: List[List[Dict]] = [[] for _ in regions]
    for w in words:
        midpoint = (w["start"] + w["end"]) / 2.0
        idx = _region_of(midpoint, regions)
        if idx is not None:
            buckets[idx].append(w)

    segments: List[Dict] = []
    MAX_WORD_GAP = 0.5

    for (region_start, region_end), members in zip(regions, buckets):
        if not members:
            continue
        members.sort(key=lambda w: w["start"])
        
        # Split into sub-buckets if word gap > MAX_WORD_GAP
        sub_buckets = []
        current_sub = [members[0]]
        
        for w in members[1:]:
            if w["start"] - current_sub[-1]["end"] > MAX_WORD_GAP:
                sub_buckets.append(current_sub)
                current_sub = [w]
            else:
                current_sub.append(w)
        if current_sub:
            sub_buckets.append(current_sub)
            
        for i, sub in enumerate(sub_buckets):
            text = " ".join(w["text"] for w in sub)
            avg_prob = sum(w["prob"] for w in sub) / len(sub)
            
            # Tightly bound the sub-bucket to the words, with a tiny pad,
            # but don't exceed the original region's boundaries
            start = max(region_start, sub[0]["start"] - 0.1)
            end = min(region_end, sub[-1]["end"] + 0.1)
            
            segments.append({
                "text": text,
                "start": start,
                "end": end,
                "avg_prob": avg_prob,
            })
    return segments
