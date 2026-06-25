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
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from azure.storage.blob import (
    BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings,
)
import requests as _req

load_dotenv()

# ── Module-level configuration ─────────────────────────────────────────────────

CLIENT_CODE         = "CLIENT002"
UNKNOWN_THRESHOLD   = float(os.getenv("VAD_UNKNOWN_THRESHOLD", "-2.0"))
DEFAULT_LANGUAGE    = os.getenv("VAD_LANGUAGE", "hi")
SEGMENT_MERGE_GAP_S    = 2.0   # merge same-speaker segments whose gap is ≤ this
MAX_SEGMENT_DURATION_S = 30.0  # never produce a segment longer than this
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

_PUNCT_RE = re.compile(r'[।॥\.,\?!;:\'"()\[\]{}\-–—/\\|@#%^&*+=<>~`]')


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

def merge_close_segments(segments: List[Dict]) -> List[Dict]:
    """Merge consecutive same-speaker segments with gap ≤ SEGMENT_MERGE_GAP_S,
    capped at MAX_SEGMENT_DURATION_S."""
    merged: List[Dict] = []
    for s in segments:
        if not merged:
            merged.append(dict(s))
            continue
        prev = merged[-1]
        gap          = s["start"] - prev["end"]
        would_be_dur = s["end"] - prev["start"]
        if (prev["speaker"] == s["speaker"]
                and gap <= SEGMENT_MERGE_GAP_S
                and would_be_dur <= MAX_SEGMENT_DURATION_S):
            prev_dur = prev["end"] - prev["start"]
            new_dur  = s["end"]   - s["start"]
            total    = prev_dur + new_dur
            prev["avg_logprob"] = (
                (prev.get("avg_logprob") or 0.0) * prev_dur +
                (s.get("avg_logprob")   or 0.0) * new_dur
            ) / total if total > 0 else 0.0
            prev["end"]  = s["end"]
            prev["text"] = (prev["text"] + " " + s["text"]).strip()
        else:
            merged.append(dict(s))
    return merged


def apply_boundary_padding(segments: List[Dict], audio_duration: float) -> List[Dict]:
    """
    Add BOUNDARY_PADDING_S to each segment boundary, but only when the adjacent
    gap is wide enough (≥ 2×pad) — i.e. the adjacent audio is silence, not speech.
    Uses original boundary values to compute gaps so modifications don't cascade.
    """
    if not segments:
        return []
    orig_starts = [s["start"] for s in segments]
    orig_ends   = [s["end"]   for s in segments]
    out = [dict(s) for s in segments]
    n = len(segments)

    for i in range(n):
        prev_end   = orig_ends[i - 1]   if i > 0   else 0.0
        next_start = orig_starts[i + 1] if i < n-1 else audio_duration

        gap_left  = orig_starts[i] - prev_end
        gap_right = next_start - orig_ends[i]

        if gap_left > 0:
            pad = min(BOUNDARY_PADDING_S, gap_left / 2.0)
            out[i]["start"] = max(prev_end, orig_starts[i] - pad)
        
        if gap_right > 0:
            pad = min(BOUNDARY_PADDING_S, gap_right / 2.0)
            out[i]["end"] = min(next_start, orig_ends[i] + pad)

    return [s for s in out if round(s["end"] - s["start"], 3) > 0]


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
    """Write per-segment 16kHz mono WAV clips. Clip filename already set in seg['audio_clip']."""
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
                "-i", source_path,
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

        # ── Step 3: Transcribe ─────────────────────────────────────────────────
        if stereo:
            from openai import OpenAI as _OAI
            groq_key = os.getenv("GROQ_API_KEY")
            transcribe_client = _OAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
            # Reuse the dual-channel energy diarization from processor.py
            from processor import transcribe_dual_channel, CLIENT_PROMPT_CONFIG
            raw_segs, detected_lang = transcribe_dual_channel(
                transcribe_client, local_path, str(tmp_dir), CLIENT_CODE, language or None,
            )
        else:
            print("Mono path — single-channel customer audio")
            import requests
            from processor import CLIENT_PROMPT_CONFIG
            
            prompt = CLIENT_PROMPT_CONFIG.get(CLIENT_CODE, "")
            detected_lang = language or "auto"
            raw_segs = []
            
            runpod_api_key = os.getenv("RUNPOD_API_KEY")
            runpod_endpoint = os.getenv("RUNPOD_WHISPER_ENDPOINT")
            runpod_success = False
            
            if runpod_api_key and runpod_endpoint:
                try:
                    # Bypass the openai Python SDK entirely to avoid the 400 Bad Request
                    url = f"https://api.runpod.ai/v2/{runpod_endpoint}/runsync"
                    headers = {
                        "Authorization": f"Bearer {runpod_api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    import base64
                    runpod_mp3 = str(local_path) + ".runpod.mp3"
                    subprocess.run(["ffmpeg", "-y", "-i", local_path, "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", runpod_mp3], check=True, capture_output=True)
                    with open(runpod_mp3, "rb") as f:
                        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                        
                    payload = {
                        "input": {
                            "audio_base64": audio_b64,
                            "model": "whisper-large-v3",
                            "response_format": "verbose_json",
                            "temperature": 0
                        }
                    }
                    if language:
                        payload["input"]["language"] = language
                    if prompt:
                        payload["input"]["initial_prompt"] = prompt
                        payload["input"]["prompt"] = prompt

                    url_run = f"https://api.runpod.ai/v2/{runpod_endpoint}/run"
                    print(f"Sending base64 JSON payload to RunPod run endpoint {runpod_endpoint}...")
                    resp = requests.post(url_run, headers=headers, json=payload, timeout=30)
                    resp.raise_for_status()
                    job_data = resp.json()
                    job_id = job_data.get("id")
                    if not job_id:
                        raise ValueError(f"RunPod returned no job ID: {job_data}")
                        
                    url_status = f"https://api.runpod.ai/v2/{runpod_endpoint}/status/{job_id}"
                    runpod_data = None
                    for _ in range(120):
                        time.sleep(5)
                        s_resp = requests.get(url_status, headers=headers, timeout=30)
                        s_resp.raise_for_status()
                        runpod_data = s_resp.json()
                        if runpod_data.get("status") == "COMPLETED":
                            break
                        elif runpod_data.get("status") == "FAILED":
                            raise ValueError(f"RunPod job failed: {runpod_data}")
                            
                    if runpod_data.get("status") != "COMPLETED":
                        raise TimeoutError("RunPod timed out.")
                        
                    result_json = runpod_data.get("output", {})
                    
                    detected_lang = result_json.get("language", detected_lang)
                    
                    for s in (result_json.get("segments", []) or []):
                        txt = (s.get("text", "") or "").strip()
                        if not txt:
                            continue
                        st  = s.get("start", 0.0) or 0.0
                        en  = s.get("end", 0.0) or 0.0
                        lp  = s.get("avg_logprob", 0.0) or 0.0
                        
                        raw_segs.append({"start": st, "end": en, "text": txt,
                                          "speaker": "Customer", "avg_logprob": lp})
                    runpod_success = True
                    print("RunPod Mono transcription successful.")
                except Exception as e:
                    print(f"RunPod Mono failed: {e}. Falling back to Groq...")
                    runpod_success = False

            if not runpod_success:
                print("Using Groq API fallback for Mono transcription...")
                from openai import OpenAI as _OAI
                groq_key = os.getenv("GROQ_API_KEY")
                groq_client = _OAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
                
                _kw = dict(
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    temperature=0,
                    prompt=prompt,
                )
                if language:
                    _kw["language"] = language
                    
                with open(local_path, "rb") as f:
                    r = groq_client.audio.transcriptions.create(file=f, **_kw)
                    
                detected_lang = getattr(r, 'language', None) or detected_lang
                for s in (getattr(r, 'segments', []) or []):
                    isd = isinstance(s, dict)
                    txt = (s.get("text") if isd else getattr(s, "text", "")).strip()
                    if not txt:
                        continue
                    st = (s.get("start") if isd else getattr(s, "start", 0.0)) or 0.0
                    en = (s.get("end") if isd else getattr(s, "end", 0.0)) or 0.0
                    lp = (s.get("avg_logprob") if isd else getattr(s, "avg_logprob", 0.0)) or 0.0
                    raw_segs.append({"start": st, "end": en, "text": txt,
                                     "speaker": "Customer", "avg_logprob": lp})

        print(f"Whisper segments: {len(raw_segs)} | Language: {detected_lang}")

        audio_dur = get_audio_duration(local_path)
        raw_segs  = apply_boundary_padding(raw_segs, audio_dur)

        # ── Step 4: Filter IVR ─────────────────────────────────────────────────
        before_ivr = len(raw_segs)
        raw_segs   = [s for s in raw_segs if not is_ivr_segment(s["text"])]
        ivr_dropped = before_ivr - len(raw_segs)
        if ivr_dropped:
            print(f"IVR filter: dropped {ivr_dropped} segment(s)")

        # ── Step 5: Merge same-speaker segments with gap ≤ 2s ─────────────────
        raw_segs = merge_close_segments(raw_segs)
        print(f"After merge: {len(raw_segs)} segment(s)")

        # ── Step 5.5: Assign orig_id AFTER merging so turns are sequential ───────
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
        # For Bajaj, we upload the raw generated zip and json, though final delivery is post-LS.
        delivery_blob = upload_zip_to_delivery(conn, zip_full, file_id)
        upload_json_to_delivery(conn, transcript_data, file_id)
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
