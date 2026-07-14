#!/usr/bin/env python3
"""
vad_processor.py — Voice as Data pipeline for Bajaj Finance / Navana.ai

CLI:
    python vad_processor.py <filename.mp3> [--language hi] [--threshold -0.4]
    python vad_processor.py --print-xml          # print LS config and exit

Flow:
  1. Download stereo audio from Azure client-intake/CLIENT002/<filename>
  2. Transcribe with Groq whisper-large-v3 (dual-channel energy diarization)
  3. Post-process: strip punctuation, digits→Hindi words, expand abbrevs
     Low-confidence segments (avg_logprob < threshold) → transcript = <UNKNOWN>
  4. Segmentation: filter IVR, merge ≤2s same-speaker gaps, 250ms boundary padding
  5. Split into per-segment 16kHz mono WAV clips via ffmpeg
  6. Upload clips to Azure processing/CLIENT002/vad_clips/<file_id>/
  7. Bundle clips + transcript JSON → ZIP → Azure client-delivery/CLIENT002/
  8. Push one Label Studio task per segment (clip URL + pre-filled transcript + speaker)
  9. After LS QA approval, export_handler (Bajaj mode — TBD) produces final delivery
"""

import os
import re
import sys
import json
import shutil
import time
import subprocess
import argparse
import math
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from azure.storage.blob import (
    BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings,
)
import requests as _req

from vad_segmenter import VadConfig, detect_speech_regions, map_words_to_regions

load_dotenv()

# ── Module-level configuration ─────────────────────────────────────────────────

CLIENT_CODE         = "CLIENT002"
UNKNOWN_THRESHOLD   = float(os.getenv("VAD_UNKNOWN_THRESHOLD", "-2.0"))
DEFAULT_LANGUAGE    = os.getenv("VAD_LANGUAGE", "hi")
SEGMENT_MERGE_GAP_S    = 0.5   # merge same-speaker segments whose gap is ≤ this
MAX_SEGMENT_DURATION_S = 15.0  # never produce a segment longer than this
BOUNDARY_PADDING_S  = 0.250  # 250ms silence padding applied to each boundary
CLIP_SAMPLE_RATE    = 8000
CLIP_CHANNELS       = 1
CLIP_CODEC          = "pcm_s16le"
LS_MODEL_VERSION    = "vad-whisper-v1"
_VAD_PROJECT_ID   = os.getenv("LABEL_STUDIO_VAD_PROJECT_ID", "9")

# ── Label Studio XML config ────────────────────────────────────────────────────
# Paste this into Label Studio → Project Settings → Labeling Interface.
# Run: python vad_processor.py --print-xml

VAD_LS_XML = """<View>
  <Header value="Bajaj Finance — Voice as Data Review"/>

  <View style="display:flex; gap:16px; align-items:flex-start; margin-bottom:12px;">
    <View style="flex:1;">
      <AudioPlus name="audio" value="$audio"/>
    </View>
    <View style="flex:2; padding:10px; background:#f4f8fc; border-radius:6px;
                 font-family:monospace; font-size:12px; white-space:pre-wrap;">
      <Text name="seg_meta" value="$seg_meta"/>
    </View>
  </View>

  <Header value="Speaker" size="5"/>
  <Choices name="speaker" toName="audio" showInLine="true" required="true">
    <Choice value="Agent"/>
    <Choice value="Customer"/>
  </Choices>

  <Header value="Transcript (Devanagari — verbatim)" size="5"/>
  <TextArea
    name="transcript"
    toName="audio"
    placeholder="Verbatim transcript in Devanagari. Use &lt;UNKNOWN&gt; for unintelligible speech."
    maxSubmissions="1"
    editable="true"
    rows="5"
    required="true"/>

  <Text name="language_tag" value="$language"/>
</View>"""

# ── IVR keyword detection ──────────────────────────────────────────────────────

_IVR_KEYWORDS = [
    "press ", "दबाएं", "दबाइए", "दबाये",
    "welcome to bajaj", "bajaj finance में आपका स्वागत",
    "please hold", "please wait", "your call is important",
    "connecting you", "कृपया प्रतीक्षा", "कृपया होल्ड",
    "for hindi", "hindi ke liye", "हिंदी के लिए",
    "for english", "press 1", "press 2",
    "ivr", "आपकी कॉल",
    "please stay on the line",
    "your call is on hold",
    "put your call on hold",
    "thank you for calling",
    "our menu options have changed",
    "for english press"
]


def is_ivr_segment(text: str) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in _IVR_KEYWORDS)


# ── Hindi number-to-words ──────────────────────────────────────────────────────
# Custom implementation — covers 0 through 99 crore (9,99,99,999).
# Indian number system: lakh = 1,00,000 | crore = 1,00,00,000

_ONES = [
    "", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह",
    "सत्रह", "अठारह", "उन्नीस",
]
_TENS = ["", "", "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे"]
_COMPOUND = {
    21: "इक्कीस", 22: "बाईस",    23: "तेईस",    24: "चौबीस",   25: "पच्चीस",
    26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस", 29: "उनतीस",
    31: "इकतीस",  32: "बत्तीस",  33: "तैंतीस",  34: "चौंतीस",  35: "पैंतीस",
    36: "छत्तीस", 37: "सैंतीस",  38: "अड़तीस",  39: "उनतालीस",
    41: "इकतालीस",42: "बयालीस",  43: "तैंतालीस",44: "चवालीस",  45: "पैंतालीस",
    46: "छियालीस",47: "सैंतालीस",48: "अड़तालीस", 49: "उनचास",
    51: "इक्यावन", 52: "बावन",   53: "तिरपन",   54: "चौवन",    55: "पचपन",
    56: "छप्पन",  57: "सत्तावन", 58: "अट्ठावन", 59: "उनसठ",
    61: "इकसठ",   62: "बासठ",    63: "तिरसठ",   64: "चौंसठ",   65: "पैंसठ",
    66: "छियासठ", 67: "सड़सठ",   68: "अड़सठ",   69: "उनहत्तर",
    71: "इकहत्तर",72: "बहत्तर",  73: "तिहत्तर", 74: "चौहत्तर", 75: "पचहत्तर",
    76: "छिहत्तर",77: "सतहत्तर", 78: "अठहत्तर", 79: "उन्यासी",
    81: "इक्यासी",82: "बयासी",   83: "तिरासी",  84: "चौरासी",  85: "पचासी",
    86: "छियासी", 87: "सत्तासी", 88: "अट्ठासी", 89: "नवासी",
    91: "इक्यानवे",92: "बानवे",  93: "तिरानवे", 94: "चौरानवे", 95: "पचानवे",
    96: "छियानवे",97: "सत्तानवे",98: "अट्ठानवे", 99: "निन्यानवे",
}


def _two_digit(n: int) -> str:
    if n == 0:
        return ""
    if n < 20:
        return _ONES[n]
    if n % 10 == 0:
        return _TENS[n // 10]
    return _COMPOUND.get(n, f"{_TENS[n // 10]} {_ONES[n % 10]}")


def _three_digit(n: int) -> str:
    if n < 100:
        return _two_digit(n)
    h = n // 100
    r = n % 100
    prefix = ("एक" if h == 1 else _ONES[h]) + " सौ"
    return prefix if r == 0 else f"{prefix} {_two_digit(r)}"


def num_to_hindi(n: int) -> str:
    if n == 0:
        return "शून्य"
    if n < 0:
        return "माइनस " + num_to_hindi(-n)
    parts = []
    if n >= 10_000_000:
        parts.append(f"{_three_digit(n // 10_000_000)} करोड़")
        n %= 10_000_000
    if n >= 100_000:
        parts.append(f"{_two_digit(n // 100_000)} लाख")
        n %= 100_000
    if n >= 1_000:
        parts.append(f"{_three_digit(n // 1_000)} हज़ार")
        n %= 1_000
    if n:
        parts.append(_three_digit(n))
    return " ".join(parts)


# Matches Indian-format numbers: single digit OR multi-digit with optional commas.
# Examples: 2, 33, 4,60,000, 10000
_NUM_RE = re.compile(r"\b\d[\d,]*\d\b|\b\d\b")


def _replace_number(m: re.Match) -> str:
    try:
        return num_to_hindi(int(m.group(0).replace(",", "")))
    except (ValueError, OverflowError):
        return m.group(0)


def digits_to_hindi_words(text: str) -> str:
    return _NUM_RE.sub(_replace_number, text)


# ── Abbreviation expansion ─────────────────────────────────────────────────────

_ABBREVS: Dict[str, str] = {
    "EMI":  "ई एम आई",
    "KYC":  "के वाई सी",
    "OTP":  "ओ टी पी",
    "SMS":  "एस एम एस",
    "NACH": "एन ए सी एच",
    "ECS":  "ई सी एस",
    "PAN":  "पी ए एन",
    "ATM":  "ए टी एम",
    "UPI":  "यू पी आई",
    "NBFC": "एन बी एफ सी",
    "RBI":  "आर बी आई",
    "PIN":  "पी आई एन",
    "GST":  "जी एस टी",
    "NOC":  "एन ओ सी",
    "PDC":  "पी डी सी",
    "SBI":  "एस बी आई",
    "CIBIL":"सी आई बी आई एल",
    "NACH": "एन ए सी एच",
    "BFL":  "बी एफ एल",
}
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_abbreviations(text: str) -> str:
    return _ABBREV_RE.sub(lambda m: _ABBREVS.get(m.group(0).upper(), m.group(0)), text)


# ── Known Whisper mis-transcriptions for Bajaj Finance domain ─────────────────
# Whisper consistently mis-hears these on Indian telephony audio.
# Applied before other post-processing so corrections feed into abbrev expansion.

_CORRECTIONS: Dict[str, str] = {
    "बजात पानस":   "बजाज फाइनेंस",
    "बजाज पनस":    "बजाज फाइनेंस",
    "बजात फाइनेंस": "बजाज फाइनेंस",
    "बजात फायनेंस": "बजाज फाइनेंस",
    "बजाज फायनेंस": "बजाज फाइनेंस",
    "पारेसमल":     "पर्सनल",
    "पारसनल":      "पर्सनल",
}


def apply_corrections(text: str) -> str:
    for wrong, right in _CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


# ── Whisper repetition loop stripper ──────────────────────────────────────────
# Whisper hallucinates by looping a short phrase at the end of a segment,
# e.g. "अब चाहिए अब चाहिए अब चाहिए अब चाहिए". Detect and truncate.

def strip_repetition_loop(text: str, min_reps: int = 3, max_window: int = 6) -> str:
    words = text.split()
    n = len(words)
    for w in range(1, min(max_window + 1, n // min_reps + 1)):
        i = 0
        while i <= n - w * min_reps:
            phrase = words[i:i + w]
            count = 1
            j = i + w
            while j + w <= n and words[j:j + w] == phrase:
                count += 1
                j += w
            if count >= min_reps:
                return " ".join(words[:i + w])
            i += 1
    return text


# ── Smart Agent/Customer speaker detection ─────────────────────────────────────
# Channel-based assumption (L=Agent) can be wrong. Cross-check by looking for
# Bajaj Finance company name in the first few segments — that speaker is the Agent.

_AGENT_SIGNALS = [
    "बजाज", "bajaj", "फाइनेंस", "finance", "नमस्कार",
    "good morning", "good afternoon", "good evening",
    "बजात", "बजाज फाइनेंस",
]


def detect_speaker_map(segments: List[Dict]) -> Dict[str, str]:
    """
    Return {raw_speaker: role} by detecting which raw speaker opened the call
    with the company greeting. Falls back to channel-order assumption.
    """
    default = {"Speaker A": "Agent", "Speaker B": "Customer"}
    for seg in segments[:4]:
        txt = seg.get("text", "").lower()
        if any(sig.lower() in txt for sig in _AGENT_SIGNALS):
            opening_speaker = seg.get("speaker", "Speaker A")
            if opening_speaker == "Speaker A":
                return {"Speaker A": "Agent", "Speaker B": "Customer"}
            else:
                print(f"Speaker swap detected — {opening_speaker} opened with company greeting → Agent")
                return {"Speaker A": "Customer", "Speaker B": "Agent"}
    print("No agent signal found in first 4 segments — using channel-order default")
    return default


# ── Punctuation stripping ──────────────────────────────────────────────────────
# Bajaj spec: no ।  , ? . or any special symbols.

_PUNCT_RE = re.compile(r'[।॥\.,\?!;:\'"()\[\]{}\-–—/\\|@#%^&*+=~`]')


def strip_punctuation(text: str) -> str:
    return _PUNCT_RE.sub("", text)


# ── Full post-processing ───────────────────────────────────────────────────────

def postprocess(text: str, avg_logprob: float, threshold: float) -> str:
    """
    Low-confidence segments → <UNKNOWN> (annotator transcribes from audio).
    High-confidence → corrections → strip repetition loops → strip punctuation
    → digits → abbrevs.
    """
    if avg_logprob < threshold:
        return "<UNKNOWN>"
    text = apply_corrections(text.strip())
    text = strip_repetition_loop(text)
    text = strip_punctuation(text)
    text = digits_to_hindi_words(text)
    text = expand_abbreviations(text)
    return re.sub(r"\s+", " ", text).strip()


# ── Timestamp helpers ──────────────────────────────────────────────────────────

def seconds_to_timecode(s: float) -> str:
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def confidence_label(avg_logprob: Optional[float], threshold: float) -> str:
    if avg_logprob is None:
        return "unknown"
    return "high" if avg_logprob >= threshold else "low"


# ── Segmentation ───────────────────────────────────────────────────────────────

def merge_and_pad_segments(segments: List[Dict]) -> List[Dict]:
    """Merge same-speaker segments with gap <= SEGMENT_MERGE_GAP_S, provided no 
    other speaker intervenes. Then apply BOUNDARY_PADDING_S to boundaries, ensuring 
    no overlaps occur between segments."""
    if not segments:
        return []

    segments.sort(key=lambda s: s["start"])
    merged: List[Dict] = []
    
    for s in segments:
        if not merged:
            merged.append(dict(s))
            continue
            
        prev = merged[-1]
        gap = s["start"] - prev["end"]
        
        # If consecutive segments are the same speaker, no other speaker spoke in between.
        if prev["speaker"] == s["speaker"] and gap <= SEGMENT_MERGE_GAP_S and (s["end"] - prev["start"]) <= MAX_SEGMENT_DURATION_S:
            prev_dur = prev["end"] - prev["start"]
            new_dur  = s["end"] - s["start"]
            total = prev_dur + new_dur
            prev["avg_logprob"] = (
                (prev.get("avg_logprob") or 0.0) * prev_dur +
                (s.get("avg_logprob") or 0.0) * new_dur
            ) / total if total > 0 else 0.0
            prev["end"] = max(prev["end"], s["end"])
            if s["text"]:
                prev["text"] = (prev["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))

    out: List[Dict] = []
    for i in range(len(merged)):
        cur = merged[i]
        prev_end = out[-1]["end"] if out else 0.0
        next_start = merged[i+1]["start"] if i < len(merged) - 1 else float('inf')
        
        start = cur["start"]
        end = cur["end"]
        
        if start < prev_end:
            start = prev_end
        if start > end:
            end = start
            
        gap_left = start - prev_end
        if gap_left > 0:
            pad = min(BOUNDARY_PADDING_S, gap_left / 2.0)
            start -= pad
            
        gap_right = next_start - end
        if gap_right > 0:
            pad = min(BOUNDARY_PADDING_S, gap_right / 2.0)
            end += pad
            
        cur["start"] = start
        cur["end"] = end
        
        if round(end - start, 3) > 0:
            out.append(cur)

    return out


def get_audio_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip() or "0")


# ── Audio clip splitting ───────────────────────────────────────────────────────

def split_clips(
    source_path: str,
    segments: List[Dict],
    clip_dir: Path,
) -> None:
    """Write per-segment 16kHz mono WAV clips. Clip filename already set in seg['audio_clip'].
    Each segment is cut from its own channel via seg['_clip_source'] (stereo), falling
    back to source_path (mono / unset) so clips stay channel-specific."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments:
        clip_path = clip_dir / seg["audio_clip"]
        dur = seg["_end_s"] - seg["_start_s"]
        if dur <= 0:
            print(f"  Skipping {seg['audio_clip']}: zero/negative duration")
            continue
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", seg.get("_clip_source") or source_path,
                "-ss", str(seg["_start_s"]),
                "-t",  str(dur),
                "-ar", str(CLIP_SAMPLE_RATE),
                "-ac", str(CLIP_CHANNELS),
                "-c:a", CLIP_CODEC,
                str(clip_path),
            ],
            check=True, capture_output=True,
        )


# ── Azure helpers ──────────────────────────────────────────────────────────────

def _parse_conn_str(conn_str: str) -> Tuple[str, str]:
    parts: Dict[str, str] = {}
    for item in conn_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            parts[k] = v
    return parts.get("AccountName", ""), parts.get("AccountKey", "")


def _make_sas(conn_str: str, container: str, blob_name: str, days: int = 30) -> str:
    account_name, account_key = _parse_conn_str(conn_str)
    token = generate_blob_sas(
        account_name=account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(days=days),
    )
    return f"https://{account_name}.blob.core.windows.net/{container}/{blob_name}?{token}"


def download_audio(filename: str) -> Tuple[str, str]:
    """Download from Azure client-intake/CLIENT002/*filename. Returns (local_path, pure_filename)."""
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set")
    svc = BlobServiceClient.from_connection_string(conn)
    cc  = svc.get_container_client("client-intake")
    blobs   = list(cc.list_blobs(name_starts_with=f"{CLIENT_CODE}/"))
    matches = [b.name for b in blobs if b.name.endswith(filename)]
    if not matches:
        raise FileNotFoundError(f"No blob in client-intake matching {CLIENT_CODE}/*{filename}")
    blob_name = sorted(matches)[-1]
    pure      = blob_name.split("/")[-1]
    tmp_dir   = Path(f"/tmp/vaidikai_vad/{uuid.uuid4().hex}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = str(tmp_dir / pure)
    with open(local_path, "wb") as f:
        f.write(svc.get_blob_client("client-intake", blob_name).download_blob().readall())
    print(f"Downloaded: {blob_name} → {local_path}")
    return local_path, pure


def upload_clip_to_processing(conn_str: str, clip_path: Path, file_id: str) -> str:
    """Upload one WAV clip to processing/CLIENT002/vad_clips/<file_id>/. Returns SAS URL."""
    blob_name = f"{CLIENT_CODE}/vad_clips/{file_id}/{clip_path.name}"
    svc = BlobServiceClient.from_connection_string(conn_str)
    bc  = svc.get_blob_client(container="processing", blob=blob_name)
    with open(clip_path, "rb") as f:
        bc.upload_blob(f, overwrite=True,
                       content_settings=ContentSettings(content_type="audio/wav"))
    return _make_sas(conn_str, "processing", blob_name)


def upload_zip_to_delivery(conn_str: str, zip_path: str, file_id: str) -> str:
    """Upload ZIP to client-delivery/CLIENT002/. Returns blob name."""
    blob_name = f"{CLIENT_CODE}/{CLIENT_CODE}_{file_id}_vad.zip"
    svc = BlobServiceClient.from_connection_string(conn_str)
    bc  = svc.get_blob_client(container="client-delivery", blob=blob_name)
    with open(zip_path, "rb") as f:
        bc.upload_blob(f, overwrite=True,
                       content_settings=ContentSettings(content_type="application/zip"))
    print(f"Uploaded ZIP → client-delivery/{blob_name}")
    return blob_name


def upload_json_to_delivery(conn_str: str, data: dict, file_id: str) -> str:
    """Upload transcript JSON to client-delivery/CLIENT002/. Returns blob name."""
    blob_name = f"{CLIENT_CODE}/{CLIENT_CODE}_{file_id}_transcript.json"
    svc = BlobServiceClient.from_connection_string(conn_str)
    bc  = svc.get_blob_client(container="client-delivery", blob=blob_name)
    bc.upload_blob(
        json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
    )
    print(f"Uploaded JSON → client-delivery/{blob_name}")
    return blob_name


# ── Label Studio push ──────────────────────────────────────────────────────────

def _ls_headers() -> dict:
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "").strip('"').strip("'")
    ls_url  = os.getenv("LABEL_STUDIO_URL", "").rstrip("/")
    if api_key.startswith("eyJ"):
        try:
            import base64 as _b64
            payload = api_key.split(".")[1]
            payload += "=" * (4 - len(payload) % 4)
            claims = json.loads(_b64.b64decode(payload))
            if claims.get("token_type") == "refresh":
                r = _req.post(f"{ls_url}/api/token/refresh",
                              json={"refresh": api_key}, timeout=15)
                r.raise_for_status()
                api_key = r.json()["access"]
        except Exception as e:
            print(f"LS token refresh failed: {e}")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}


def push_to_label_studio(
    segments: List[Dict],
    clip_sas_urls: List[Optional[str]],
    source_file: str,
    language: str,
    project_id: str,
) -> Dict[str, Any]:
    """
    Push one LS task per segment.
    Each task: segment audio clip + pre-filled speaker choice + pre-filled transcript.
    """
    ls_url  = os.getenv("LABEL_STUDIO_URL", "").rstrip("/")
    headers = _ls_headers()

    if not ls_url:
        return {"status": "error", "error": "LABEL_STUDIO_URL not set"}

    task_ids: List[int] = []
    failed = 0

    for seg, sas_url in zip(segments, clip_sas_urls):
        if not sas_url:
            print(f"  Seg {seg['segment_id']}: no clip URL — skipping LS push")
            failed += 1
            continue

        seg_meta = (
            f"Seg {seg['segment_id']:03d}  |  "
            f"{seg['start_time']} → {seg['end_time']}  |  "
            f"Duration: {seg['duration_seconds']:.2f}s  |  "
            f"Confidence: {seg['confidence']}  |  "
            f"Source: {source_file}"
        )

        task_payload = {
            "data": {
                "audio":        sas_url,
                "seg_meta":     seg_meta,
                "language":     f"Language: {language}",
                "filename":     source_file,
                "client_code":  CLIENT_CODE,
                "segment_id":   seg["segment_id"],
                "start_time":   seg["start_time"],
                "end_time":     seg["end_time"],
                "audio_clip":   seg["audio_clip"],
            }
        }

        predictions_result = [
            {
                "from_name": "speaker",
                "to_name":   "audio",
                "type":      "choices",
                "value":     {"choices": [seg["speaker"]]},
            },
            {
                "from_name": "transcript",
                "to_name":   "audio",
                "type":      "textarea",
                "value":     {"text": [seg["transcript"]]},
            },
        ]

        # Create task via /tasks (returns full task object with id).
        # /import is faster for bulk but returns only counts — no task IDs,
        # so predictions can't be posted. /tasks is one request per segment
        # but gives us the id we need immediately.
        r = _req.post(
            f"{ls_url}/api/tasks",
            json={**task_payload, "project": int(project_id)},
            headers=headers, timeout=30,
        )
        if not r.ok:
            print(f"  Seg {seg['segment_id']} task create failed: {r.status_code} {r.text[:120]}")
            failed += 1
            continue

        task_id: Optional[int] = None
        try:
            resp = r.json()
            task_id = resp.get("id") if isinstance(resp, dict) else None
        except Exception:
            pass

        if task_id:
            task_ids.append(task_id)
            pr = _req.post(
                f"{ls_url}/api/predictions",
                json={"task": task_id, "result": predictions_result,
                      "model_version": LS_MODEL_VERSION},
                headers=headers, timeout=20,
            )
            if not pr.ok:
                print(f"  Seg {seg['segment_id']} predictions warn: {pr.status_code}")
        else:
            print(f"  Seg {seg['segment_id']}: created but no task_id returned")

    print(f"LS push complete: {len(task_ids)} tasks created, {failed} failed.")
    return {"status": "success", "task_ids": task_ids, "failed": failed}


# ── VAD-first transcription (word-level) ───────────────────────────────────────
# Design 2026-06-30: Silero VAD defines segment boundaries; each channel is
# transcribed once with word-level timestamps; words map into VAD windows by
# midpoint. Silence has no window, so no word lands there → silence-free segments.
# Replaces Whisper-default segmentation for the Bajaj / CLIENT002 flow.

_VAD_CFG = VadConfig()


def split_stereo_channels(local_path: str, tmp_dir: Path) -> Tuple[str, str]:
    """Split stereo into two 16kHz mono channel WAVs (L, R). Channel == speaker."""
    ch_dir = tmp_dir / "ch"
    ch_dir.mkdir(parents=True, exist_ok=True)
    left  = str(ch_dir / "L.wav")
    right = str(ch_dir / "R.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(local_path),
        "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[L][R]",
        "-map", "[L]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", left,
        "-map", "[R]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", right,
    ], check=True, capture_output=True)
    return left, right


def _parse_words(segments: List[Dict], top_words: List[Dict]) -> List[Dict]:
    """Normalise word objects from a verbose_json response into
    {text, start, end, prob}. Words may live per-segment ('words') or top-level.
    prob falls back to exp(segment avg_logprob), then 1.0."""
    out: List[Dict] = []

    def _emit(w: Dict, seg_lp: Optional[float]) -> None:
        txt = (w.get("word") or w.get("text") or "").strip()
        st, en = w.get("start"), w.get("end")
        if not txt or st is None or en is None:
            return
        prob = w.get("probability", w.get("prob"))
        if prob is None:
            prob = math.exp(seg_lp) if seg_lp is not None else 1.0
        out.append({"text": txt, "start": float(st), "end": float(en), "prob": float(prob)})

    for s in (segments or []):
        seg_lp = s.get("avg_logprob")
        for w in (s.get("words") or []):
            _emit(w, seg_lp)
    if not out:
        for w in (top_words or []):
            _emit(w, None)
    return out


def transcribe_words(path: str, language: Optional[str]) -> Tuple[List[Dict], str]:
    """Transcribe one channel WAV with WORD-level timestamps. RunPod primary
    (timestamp_granularities=word), Groq fallback. Returns (words, detected_lang)
    where words = [{text, start, end, prob}, …] on the file's own timeline.

    NOTE: the RunPod faster-whisper-server's word-timestamp support must be
    verified on the live endpoint — if it returns no words, this falls back to
    Groq (sample phase). Vultr production must enable word_timestamps natively."""
    detected = language or "auto"
    raw_url = os.getenv("RAW_RUNPOD_URL")

    if raw_url:
        try:
            if "runsync" in raw_url or "api.runpod.ai/v2" in raw_url:
                import base64
                with open(path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                
                payload = {
                    "input": {
                        "endpoint": "/v1/audio/transcriptions",
                        "file": audio_b64,
                        "model": "Systran/faster-whisper-large-v3",
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["word"],
                        "temperature": 0.0,
                        "vad_filter": True
                    }
                }
                if language:
                    payload["input"]["language"] = language
                
                headers = {"Content-Type": "application/json"}
                api_key = os.getenv("RUNPOD_API_KEY")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                resp = _req.post(raw_url, json=payload, headers=headers, timeout=600)
                resp.raise_for_status()
                # Runpod serverless wrapper returns output inside 'output' key
                rj = resp.json()
                if "output" in rj:
                    rj = rj["output"]
            else:
                with open(path, "rb") as f:
                    data = {
                        "model": "Systran/faster-whisper-large-v3",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "word",
                        "temperature": "0.0",
                        "vad_filter": "true",
                    }
                    if language:
                        data["language"] = language
                    resp = _req.post(raw_url, files={"file": f}, data=data, timeout=600)
                    resp.raise_for_status()
                    rj = resp.json()

            words = _parse_words(rj.get("segments", []), rj.get("words", []))
            if words:
                return words, rj.get("language", detected)
            print("RunPod returned no word timestamps — falling back to Groq for words.")
        except Exception as e:
            print(f"RunPod word transcription failed: {e}. Falling back to Groq...")

    # Groq fallback — word granularity.
    from openai import OpenAI as _OAI
    groq = _OAI(api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1")
    _kw = dict(model="whisper-large-v3", response_format="verbose_json",
               timestamp_granularities=["word"], temperature=0)
    if language:
        _kw["language"] = language
    with open(path, "rb") as f:
        r = groq.audio.transcriptions.create(file=f, **_kw)
    detected = getattr(r, "language", None) or detected

    def _word_dict(w: Any) -> Dict:
        """Normalise a single SDK word object to a plain dict so _emit().get()
        works and confidence (probability) is preserved, not dropped."""
        if isinstance(w, dict):
            return w
        return {
            "word": getattr(w, "word", None),
            "text": getattr(w, "text", None),
            "start": getattr(w, "start", None),
            "end": getattr(w, "end", None),
            "probability": getattr(w, "probability", None),
            "prob": getattr(w, "prob", None),
        }

    def _seg_dict(s: Any) -> Dict:
        """Normalise a segment, converting its nested SDK word objects to dicts."""
        if isinstance(s, dict):
            return s
        return {
            "avg_logprob": getattr(s, "avg_logprob", None),
            "words": [_word_dict(w) for w in (getattr(s, "words", None) or [])],
        }

    segs = [_seg_dict(s) for s in (getattr(r, "segments", []) or [])]
    top  = [_word_dict(w) for w in (getattr(r, "words", []) or [])]
    words = _parse_words(segs, top)
    return words, detected


def vad_first_channel(
    channel_wav: str, speaker_label: str,
    language: Optional[str], cfg: VadConfig = _VAD_CFG,
) -> Tuple[List[Dict], str]:
    """One channel → Silero VAD regions + word transcription + midpoint map.
    Returns segments in the downstream schema
    [{start, end, text, speaker, avg_logprob}] and the detected language."""
    regions = detect_speech_regions(channel_wav, cfg)
    words, lang = transcribe_words(channel_wav, language)
    segs: List[Dict] = []
    for m in map_words_to_regions(words, regions):
        # map gives avg_prob (0–1); downstream <UNKNOWN> logic expects avg_logprob.
        segs.append({
            "start": m["start"], "end": m["end"], "text": m["text"],
            "speaker": speaker_label,
            "avg_logprob": math.log(max(m["avg_prob"], 1e-9)),
            # Clip from THIS channel's isolated audio, not the mixed file.
            "_clip_source": channel_wav,
        })
    return segs, lang


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_vad(
    filename: str,
    language: str = DEFAULT_LANGUAGE,
    threshold: float = UNKNOWN_THRESHOLD,
    project_id: str = _VAD_PROJECT_ID,
) -> Dict[str, Any]:
    """
    Full Voice as Data pipeline for one Bajaj Finance audio file.
    Returns a summary dict with status, segment count, delivery paths, LS task IDs.
    """
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set")

    print(f"\n{'='*62}")
    print(f"VOICE AS DATA — {filename}")
    print(f"Language: {language} | UNKNOWN threshold: {threshold} | LS project: {project_id}")
    print(f"{'='*62}")

    # ── Step 1: Download ───────────────────────────────────────────────────────
    local_path, pure_filename = download_audio(filename)
    file_id = Path(pure_filename).stem          # e.g. "8b6f13e3" from "8b6f13e3.mp3"
    
    # Strip leading timestamp from file_id if present (e.g., 20260609_131001_)
    import re
    file_id = re.sub(r"^\d{8}_\d{6}_", "", file_id)
    
    tmp_dir = Path(local_path).parent

    # Extract audio via ffmpeg before processing
    extracted_wav = str(tmp_dir / f"{file_id}_extracted.wav")
    print(f"Extracting {local_path} -> {extracted_wav}")
    subprocess.run(
        ["ffmpeg", "-y", "-i", local_path, "-vn", "-acodec", "pcm_s16le", extracted_wav],
        check=True, capture_output=True
    )
    local_path = extracted_wav

    try:
        # ── Step 2: Detect stereo vs mono ─────────────────────────────────────
        pc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels",
             "-of", "default=noprint_wrappers=1:nokey=1", local_path],
            check=True, capture_output=True, text=True,
        )
        stereo = int((pc.stdout.strip() or "1")) >= 2
        print(f"Audio: {'STEREO — channel=speaker' if stereo else 'MONO — gap-based diarization'}")

        # ── Step 3: Transcribe — VAD-first, word-level (design 2026-06-30) ─────
        # Silero VAD defines segment boundaries; each channel is transcribed once
        # with word-level timestamps; words map into VAD windows by midpoint.
        # Silence has no window → no word lands there → silence-free segments.
        if stereo:
            print("Stereo — per-channel Silero VAD + word-level transcription")
            left, right = split_stereo_channels(local_path, tmp_dir)
            a_segs, a_lang = vad_first_channel(left,  "Speaker A", language or None)
            b_segs, b_lang = vad_first_channel(right, "Speaker B", language or None)
            # Keep L/R on disk until clips are cut — each segment is clipped from
            # its own channel (_clip_source). tmp_dir cleanup removes them later.
            raw_segs = sorted(a_segs + b_segs, key=lambda s: s["start"])
            if language:
                detected_lang = language
            else:
                a_chars = sum(len(s["text"]) for s in a_segs)
                b_chars = sum(len(s["text"]) for s in b_segs)
                detected_lang = a_lang if a_chars >= b_chars else b_lang
        else:
            # Mono has no channel separation, so speaker identity cannot be derived
            # here — label "Unknown" rather than asserting "Customer" (which would
            # mislabel any agent speech). The LS annotator sets the speaker; export
            # then keeps only customer segments.
            print("Mono — single-channel Silero VAD + word-level transcription (speaker Unknown)")
            raw_segs, detected_lang = vad_first_channel(
                local_path, "Unknown", language or None,
            )

        print(f"VAD-first segments: {len(raw_segs)} | Language: {detected_lang}")

        # Merge close same-speaker segments and apply boundary padding globally
        # to ensure no overlapping segments between speakers.
        raw_segs = merge_and_pad_segments(raw_segs)

        # ── Step 4: Filter IVR ─────────────────────────────────────────────────
        before_ivr = len(raw_segs)
        raw_segs   = [s for s in raw_segs if not is_ivr_segment(s["text"])]
        ivr_dropped = before_ivr - len(raw_segs)
        if ivr_dropped:
            print(f"IVR filter: dropped {ivr_dropped} segment(s)")

        # ── Step 5: Assign orig_id so turns are sequential ─────────────────────
        for i, s in enumerate(raw_segs, 1):
            s["orig_id"] = i

        # ── Step 6: Build final segment records ────────────────────────────────
        # Detect Agent vs Customer by content (company name in opening),
        # not just channel order — recording layout varies per file.
        SPEAKER_MAP = detect_speaker_map(raw_segs)

        clip_dir = tmp_dir / f"{CLIENT_CODE}_{file_id}"
        clip_dir.mkdir(exist_ok=True)
        audio_segments_dir = clip_dir / "audio_segments"
        audio_segments_dir.mkdir(exist_ok=True)

        final_segments: List[Dict] = []
        for s in raw_segs:
            orig_i   = s.get("orig_id", 0)
            speaker  = SPEAKER_MAP.get(s["speaker"], s["speaker"])
            lp       = s.get("avg_logprob") or 0.0
            processed = postprocess(s["text"], lp, threshold)
            duration  = round(s["end"] - s["start"], 3)
            clip_name = f"{file_id}_segment_{orig_i}.wav"

            final_segments.append({
                # Delivery fields (go in JSON)
                "segment_id":      orig_i,
                "speaker":         speaker,
                "start_time":      seconds_to_timecode(s["start"]),
                "end_time":        seconds_to_timecode(s["end"]),
                "duration_seconds": duration,
                "audio_clip":      clip_name,
                "transcript":      processed,
                "language":        (detected_lang or language or "hi").capitalize(),
                "confidence":      confidence_label(lp, threshold),
                "source_file":     pure_filename,
                # Internal only (stripped from delivery JSON)
                "_start_s":        s["start"],
                "_end_s":          s["end"],
                "_avg_logprob":    lp,
                "_clip_source":    s.get("_clip_source"),
            })

        print(f"Final segments: {len(final_segments)}")

        # ── Step 8: Split per-segment audio clips ──────────────────────────────
        print("Splitting audio clips (ffmpeg)...")
        split_clips(local_path, final_segments, audio_segments_dir)
        print(f"Clips in: {audio_segments_dir}")

        # ── Step 9: Upload clips to Azure processing/ → SAS URLs for LS ───────
        print("Uploading clips to Azure...")
        clip_sas_urls: List[Optional[str]] = []
        for seg in final_segments:
            clip_path = audio_segments_dir / seg["audio_clip"]
            if clip_path.exists():
                sas = upload_clip_to_processing(conn, clip_path, file_id)
                clip_sas_urls.append(sas)
            else:
                print(f"  Warning: clip not found: {clip_path.name}")
                clip_sas_urls.append(None)
        uploaded_count = sum(1 for u in clip_sas_urls if u)
        print(f"Uploaded {uploaded_count}/{len(final_segments)} clips")

        # ── Step 10: Build delivery JSON (public fields only) ──────────────────
        call_date = ""
        m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_", pure_filename)
        if m:
            call_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}"
            
        output_shared = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        transcript_data = []
        for seg in final_segments:
            clean_filename = re.sub(r'^\d{8}_\d{6}_', '', pure_filename)
            transcript_data.append({
                "Language Code": "hi",
                "File Name": clean_filename,
                "Segment No.": seg["segment_id"],
                "Segment File": f"audio_segments/{seg['audio_clip']}",
                "Transcription without labels": seg["transcript"],
                "Call Date (Date and Time Stamp)": call_date,
                "Output Shared (Date and Time Stamp)": output_shared
            })

        # ── Step 11: ZIP clips + JSON ──────────────────────────────────────────
        json_filename = "transcript.json"
        with open(clip_dir / json_filename, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        zip_name_stem = f"hi_{datetime.now().strftime('%Y%m%d')}_{file_id}"
        zip_base = str(tmp_dir / zip_name_stem)
        shutil.make_archive(zip_base, "zip", str(clip_dir))
        zip_full = zip_base + ".zip"
        print(f"ZIP: {zip_full}  ({os.path.getsize(zip_full) // 1024} KB)")

        # ── Step 12: Upload ZIP + JSON to client-delivery ─────────────────────
        # For Bajaj, we DO NOT upload the raw generated zip and json to delivery.
        # They will be pushed to Label Studio for review, and exported to delivery later.
        delivery_blob = "pending_review"
        # ── Step 13: Push to Label Studio ──────────────────────────────────────
        print("Pushing segments to Label Studio...")
        ls_result = push_to_label_studio(
            final_segments, clip_sas_urls,
            pure_filename, detected_lang or language, project_id,
        )

        return {
            "status":           "success",
            "file_id":          file_id,
            "source_file":      pure_filename,
            "total_segments":   len(final_segments),
            "ivr_dropped":      ivr_dropped,
            "language":         detected_lang or language,
            "threshold_used":   threshold,
            "delivery_zip":     f"client-delivery/{delivery_blob}",
            "ls_tasks_created": ls_result.get("task_ids", []),
            "ls_failed":        ls_result.get("failed", 0),
        }

    finally:
        try:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
        except Exception:
            pass


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Voice as Data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vad_processor.py call_001.mp3
  python vad_processor.py call_001.mp3 --language hi --threshold -0.4
  python vad_processor.py call_001.mp3 --ls-project 10
  python vad_processor.py --print-xml
        """,
    )
    parser.add_argument(
        "filename", nargs="?",
        help="Audio file in Azure client-intake/CLIENT002/ (e.g. call.mp3)",
    )
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE,
        help=f"Whisper language code (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--threshold", type=float, default=UNKNOWN_THRESHOLD,
        help=f"avg_logprob threshold for <UNKNOWN> tag (default: {UNKNOWN_THRESHOLD})",
    )
    parser.add_argument(
        "--ls-project", default=_VAD_PROJECT_ID,
        help=f"Label Studio project ID (default: {_VAD_PROJECT_ID})",
    )
    parser.add_argument(
        "--print-xml", action="store_true",
        help="Print the Label Studio XML config for this project and exit",
    )
    args = parser.parse_args()

    if args.print_xml:
        print(VAD_LS_XML)
        sys.exit(0)

    if not args.filename:
        parser.error("filename is required (or use --print-xml)")

    result = process_vad(
        args.filename,
        language=args.language,
        threshold=args.threshold,
        project_id=args.ls_project,
    )

    print(f"\n{'='*62}")
    print("RESULT:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
