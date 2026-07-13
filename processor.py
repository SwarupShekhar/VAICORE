import os
import json
import time
import subprocess
import tempfile
import shutil
import base64
import requests
import gc
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from azure.storage.blob import BlobServiceClient
from openai import OpenAI
from dotenv import load_dotenv
import logging
from logger import get_logger

load_dotenv()

log = get_logger("vaidikai.processor")

# Silence Azure SDK HTTP request/response INFO logging — it floods docker logs
# and buries real transcription/diarization output.
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# Global lock to ensure only one heavy transcription task runs at a time
_PROCESSING_LOCK = threading.Lock()

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)

# Client-specific role labels (Overridable)
# Format: { 'CLIENT_CODE': {'Agent': 'AgentLabel', 'Customer': 'CustomerLabel'} }
CLIENT_ROLE_CONFIG = {
    'DEFAULT': {'Agent': 'Speaker 1', 'Customer': 'Speaker 2'},
}

CLIENT_PROMPT_CONFIG = {
    # Intentionally empty. A prompt biases Whisper's output script/language.
    # Empty prompt => Whisper auto-detects the spoken language and transcribes
    # in that language's own native script (Hindi->Devanagari, English->Latin,
    # Hinglish as spoken). Do NOT add Devanagari or Latin domain phrases here.
    'DEFAULT': "",
    'CLIENT002': "नमस्कार, बजाज फायनान्स (Bajaj Finance), ईएमआय (EMI), लोन, हप्ता, वस्तू खरेदी, होय, नाही, ठीक आहे.",
}

# Rule-based tagging rules
INTENT_RULES = {
    'Greeting': ['namaskar', 'hello', 'sahayta', 'kis prakar', 'namaskar', 'hello'],
    'Product Enquiry': ['credit card', 'kharidna', 'apply', 'card chahiye', 'credit card', 'apply'],
    'Verification': ['nam', 'salary', 'designation', 'company', 'bank account', 'salary', 'name', 'company'],
    'Annual Fee Discussion': ['annual charge', 'do hazaar', 'fees', 'annual', 'fee', 'charge'],
    'Benefit Explanation': ['benefit', 'cashback', 'launch', 'reward', 'cashback', 'lounge', 'benefit', 'offer'],
    'Objection': ['nahin bharna', 'nahin hai', 'no sir', 'cannot', 'nahin'],
    'Clarification Request': ['matlab', 'kaise', 'kab', 'kahan', 'meaning', 'how', 'when'],
    'Confirmation': ['oke sar', 'han ji', 'thik hai', 'okay', 'yes sir', 'confirmed'],
    'Call Issue': ['hello', 'avaj', 'hello', 'can you hear', 'network'],
}

SENTIMENT_RULES = {
    'Positive': ['oke sar', 'han ji', 'thik hai', 'benefit', 'okay', 'yes', 'good', 'great', 'reward'],
    'Negative': ['nahin bharna', 'nahin hai', 'no sir', 'cannot', 'not', 'nahin'],
    'Curious': ['matlab', 'kaise', 'kab', 'kya', 'what', 'how', 'when', 'really'],
    'Frustrated': ['hello', 'avaj nahin', 'hello hello', 'not working', 'again'],
}

OUTCOME_RULES = {
    'Call Opened': ['namaskar', 'sahayta', 'namaskar', 'help you'],
    'Lead Identified': ['credit card', 'kharidna', 'credit card', 'want to buy'],
    'KYC In Progress': ['nam', 'salary', 'company', 'name', 'salary', 'company'],
    'Fee Objection Raised': ['annual charge', 'nahin bharna', 'annual charge', 'not pay'],
    'Objection Handled': ['lekin', 'benefit', 'reward', 'but', 'benefit', 'reward'],
    'Benefit Communicated': ['cashback', 'launch', 'reward', 'cashback', 'lounge', 'offer'],
    'Customer Clarifying': ['matlab', 'really', 'meaning'],
    'Customer Interested': ['cashback', '5%', 'cashback', 'lounge', 'interested'],
    'Call Disruption': ['hello', 'avaj', 'hello hello'],
}

def tag(text: str, rules: Dict[str, List[str]], default: str = 'Other') -> str:
    """Apply rule-based tagging to text."""
    text_lower = text.lower()
    for label, keywords in rules.items():
        if any(kw.lower() in text_lower for kw in keywords):
            return label
    return default

def get_key_signal(text: str, max_words: int = 8) -> str:
    """Extract first N words as key signal."""
    words = text.split()
    return ' '.join(words[:max_words])

def calculate_qa_status(confidence: float) -> str:
    """Determine QA status based on confidence."""
    if confidence >= 65:
        return 'HIGH CONFIDENCE'
    elif confidence >= 40:
        return 'REVIEW NEEDED'
    else:
        return 'LOW CONFIDENCE'

def filter_segment(text: str, avg_logprob: float) -> bool:
    """Keep every non-empty segment. This is a pre-annotation tool — completeness
    beats precision; annotators correct text. Dropping on avg_logprob silently
    deletes real but quiet/accented speech (caused missing 59s-1:20 spans)."""
    return bool(text and text.strip())

def identify_speaker_roles(segments: List[Dict], client_code: str) -> Dict[str, str]:
    """
    Use GPT-4o to identify which speaker ID corresponds to which role.
    """
    if not segments:
        return {}
    
    # 1. Get unique speaker IDs from the diarized output
    speaker_ids = sorted(list(set(s.get('speaker', 'Unknown') for s in segments)))
    speaker_ids = [sid for sid in speaker_ids if sid != 'Unknown']
    
    if len(speaker_ids) == 0:
        return {}
    
    if len(speaker_ids) == 1:
        # Only one speaker detected — keep raw ID so labelstudio_client maps it by order
        return {speaker_ids[0]: speaker_ids[0]}

    # 2. Prepare sample transcript (first 15 segments)
    sample_lines = [f"{s.get('speaker')}: {s.get('text')}" for s in segments[:15]]
    sample = "\n".join(sample_lines)
    
    # 3. Get labels
    config = CLIENT_ROLE_CONFIG.get(client_code, CLIENT_ROLE_CONFIG['DEFAULT'])
    agent_label = config.get('Agent', 'Agent')
    customer_label = config.get('Customer', 'Customer')

    try:
        log.info(f"Identifying speaker roles for {client_code} using AI context...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": f"Analyze the call transcript and map speaker IDs to roles. Return JSON: {{'SpeakerID': '{agent_label}', 'SpeakerID': '{customer_label}'}}"
                },
                {
                    "role": "user", 
                    "content": f"Speaker IDs: {speaker_ids}\nSample:\n{sample}"
                }
            ],
            response_format={"type": "json_object"}
        )
        mapping = json.loads(response.choices[0].message.content)
        log.info(f"AI Role Mapping: {mapping}")
        return mapping
    except Exception as e:
        log.error(f"Role identification failed: {e}")
        return {sid: sid for sid in speaker_ids}

def get_call_intelligence(transcript: str) -> Dict[str, Any]:
    """
    Perform deep strategic analysis of a call transcript using GPT-4o.
    Detects friction, latency, disputes, and self-service failures.
    """
    try:
        log.info("Running AI Intelligence Analysis on transcript...")
        response = client.chat.completions.create(
            model="gpt-4o",
            timeout=5.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Call Intelligence Analyst for a Financial Institution. "
                        "Analyze the transcript and identify the following signals:\n"
                        "1. Onboarding Friction: e-KYC, V-KYC, Net Banking, App UX issues.\n"
                        "2. Operational Pain: TAT/Disbursement delays, policy confusion.\n"
                        "3. Financial Disputes: Insurance add-ons, foreclosure fees, bouncing penalties.\n"
                        "4. Self-Service Failure: Borrowers leaving App/IVR for simple tasks (Statements, Interest Certs).\n"
                        "5. Customer Mood: Satisfied, Neutral, Frustrated, Angry.\n"
                        "6. Churn Risk: High Risk, Medium Risk, Low Risk.\n"
                        "Return ONLY a JSON object with these keys: intent (string), onboarding_friction (list), operational_pain (list), "
                        "financial_disputes (list), service_leakage (list), mood (string), churn_risk (string), summary (string)."
                    )
                },
                {"role": "user", "content": f"Transcript:\n{transcript}"}
            ],
            response_format={"type": "json_object"}
        )
        intelligence = json.loads(response.choices[0].message.content)
        return intelligence
    except Exception as e:
        log.error(f"AI Intelligence Analysis failed: {e}")
        return {
            "onboarding_friction": [], "operational_pain": [], "financial_disputes": [], 
            "service_leakage": [], "mood": "Neutral", "churn_risk": "Low Risk", "summary": ""
        }

def transcribe_dual_channel(groq_client, local_audio_path, base_temp_dir, client_code, language=None):
    """Stereo call: transcribe EACH channel independently (channel == speaker).

    Telephony stereo carries the agent on one channel, the customer on the
    other. Transcribing the MIXED downmix causes cross-talk -> on noisy 8kHz
    Whisper collapses into catastrophic repetition ("alo alo alo ..."). Verified
    on real CLIENT002/Bajaj audio: mixed downmix = pure garbage; per-channel =
    clean, coherent conversation.

    So we transcribe L and R in isolation. This also gives perfect diarization
    for free (channel identity == speaker identity) — no energy guessing, no
    pyannote. vad_filter drops the silent stretches (while the other party
    talks) that would otherwise hallucinate.

    NOTE: residual same-token repetition loops are driven by the SERVER's baked
    condition_on_previous_text=True (NOT settable over HTTP — the endpoint
    silently drops the field). Fix fully by redeploying faster-whisper-server
    with condition_on_previous_text=False and no_repeat_ngram_size=3. Client-
    side we (a) drop the Marathi prompt bias — it corrupts non-Marathi calls,
    and Bajaj runs many languages; auto-detect per channel instead — and
    (b) collapse consecutive identical hallucinated segments.

    Returns (temp_segments, detected_language). temp_segments: list of
    {start, end, text, speaker, avg_logprob} sorted by start.
    """
    import subprocess as _sp

    raw_url = os.getenv("RAW_RUNPOD_URL")

    def _collapse_repeats(text):
        """Collapse adjacent identical comma-delimited tokens — loop spam like
        'வணக்கம், வணக்கம், வணக்கம், ...' becomes 'வணக்கம்'. Readability only;
        does NOT recover the words the server looped over instead of decoding.
        Real fix = server condition_on_previous_text=False + no_repeat_ngram_size."""
        toks = [t.strip() for t in text.split(',')]
        out = []
        for t in toks:
            if not t or (out and out[-1] == t):
                continue
            out.append(t)
        return ', '.join(out)

    def _one_request(path, lang):
        """One transcription request for an audio file. RunPod primary, Groq
        fallback. Returns (segments, detected_lang)."""
        segs = []
        dlang = lang or "auto"
        if raw_url:
            try:
                import requests
                # Endpoint accepts ONLY: file, model, language, prompt,
                # response_format, temperature, timestamp_granularities, stream,
                # hotwords, vad_filter. No prompt/hotwords on purpose — domain
                # bias in the wrong language seeds hallucination loops.
                with open(path, "rb") as f:
                    data = {
                        "model": "Systran/faster-whisper-large-v2",
                        "response_format": "verbose_json",
                        "timestamp_granularities": "segment",
                        "temperature": "0.0",
                        "vad_filter": "true",
                    }
                    if lang:
                        data["language"] = lang
                    resp = requests.post(raw_url, files={"file": f}, data=data, timeout=600)
                    resp.raise_for_status()
                    rj = resp.json()
                dlang = rj.get("language", dlang)
                for s in (rj.get("segments", []) or []):
                    txt = (s.get("text", "") or "").strip()
                    if txt:
                        segs.append({
                            "start": s.get("start", 0.0) or 0.0,
                            "end": s.get("end", 0.0) or 0.0,
                            "text": txt,
                            "avg_logprob": s.get("avg_logprob", 0.0) or 0.0,
                        })
                return segs, dlang
            except Exception as e:
                log.error(f"RunPod channel transcription failed: {e}. Falling back to Groq...")
                segs = []

        # Fallback: Groq SDK
        _kw = dict(
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            temperature=0,  # anti-hallucination
        )
        if lang:
            _kw["language"] = lang
        with open(path, "rb") as f:
            r = groq_client.audio.transcriptions.create(file=f, **_kw)
        dlang = getattr(r, "language", None) or dlang
        for s in (getattr(r, "segments", []) or []):
            isd = isinstance(s, dict)
            txt = ((s.get("text") if isd else getattr(s, "text", "")) or "").strip()
            if not txt:
                continue
            segs.append({
                "start": (s.get("start") if isd else getattr(s, "start", 0.0)) or 0.0,
                "end": (s.get("end") if isd else getattr(s, "end", 0.0)) or 0.0,
                "text": txt,
                "avg_logprob": (s.get("avg_logprob") if isd else getattr(s, "avg_logprob", 0.0)) or 0.0,
            })
        return segs, dlang

    def _loop_score(segs):
        """Distinct-word ratio over a channel's text. ~0.5+ = healthy speech;
        a repetition loop ('வணக்கம் வணக்கம் ...') drives it toward 0. Returns
        1.0 for too-short input (nothing to judge)."""
        import re as _re
        words = []
        for s in segs:
            words += _re.findall(r'\S+', s.get("text", ""))
        if len(words) < 12:
            return 1.0
        return len(set(words)) / len(words)

    def _transcribe_file(path, lang):
        """Transcribe one channel. Try whole-file first (best quality). If the
        server's condition_on_previous_text=True drove it into a repetition loop
        (low distinct-word ratio), re-transcribe in independent ~25s chunks —
        each request resets context, breaking the loop. Chunk language is decided
        by majority vote across chunks (not whole-file detection, which a 'Hello'
        opener poisons to 'en'), and outlier chunks are re-transcribed pinned to
        the winner. INTERIM hack until the server is redeployed with the loop fix."""
        import math as _math
        LOOP_THRESHOLD = 0.25
        CHUNK = 25.0

        segs, dlang = _one_request(path, lang)
        if _loop_score(segs) >= LOOP_THRESHOLD:
            return segs, dlang

        log.warning(f"Repetition loop on {Path(path).name} "
                    f"(distinct-word ratio {_loop_score(segs):.2f}); re-transcribing chunked.")
        try:
            dur = float(_sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", path],
                check=True, capture_output=True, text=True).stdout.strip())
        except Exception:
            return segs, dlang

        # Transcribe each chunk auto-detecting its own language (when no upload
        # language is pinned), then MAJORITY-VOTE the language weighted by text
        # length and re-transcribe only the OUTLIER chunks pinned to the winner.
        # This beats both failure modes: "lock to whole-file" poisons every chunk
        # with the 'en' a "Hello" opener triggers; "every chunk free" lets noisy
        # chunks drift to Korean/Turkish/etc. The vote anchors to the language the
        # call is actually in; outlier re-transcription rescues the drifters.
        chunks = []
        votes = {}
        for i in range(_math.ceil(dur / CHUNK)):
            st = i * CHUNK
            cpath = f"{path}.c{i}.wav"
            try:
                _sp.run(["ffmpeg", "-y", "-ss", str(st), "-t", str(CHUNK),
                         "-i", path, "-ar", "16000", "-ac", "1",
                         "-c:a", "pcm_s16le", cpath],
                        check=True, capture_output=True)
                cs, clang = _one_request(cpath, lang)  # lang None -> per-chunk auto
                chunks.append({"path": cpath, "start": st, "segs": cs, "lang": clang})
                if clang:
                    votes[clang] = votes.get(clang, 0) + sum(len(x["text"]) for x in cs)
            except Exception as e:
                log.error(f"Chunk {i} failed: {e}")
                chunks.append({"path": cpath, "start": st, "segs": [], "lang": None})

        winner = lang or (max(votes, key=votes.get) if votes else dlang)

        # Re-transcribe chunks whose detected language disagrees with the winner.
        if not lang and winner:
            for c in chunks:
                if c["segs"] and c["lang"] != winner:
                    log.info(f"Chunk @{c['start']:.0f}s detected '{c['lang']}' != "
                             f"'{winner}'; re-transcribing pinned to '{winner}'.")
                    try:
                        cs, _ = _one_request(c["path"], winner)
                        c["segs"] = cs
                    except Exception as e:
                        log.error(f"Re-transcribe chunk @{c['start']:.0f}s failed: {e}")

        chunk_segs = []
        for c in chunks:
            for x in c["segs"]:
                x["start"] += c["start"]
                x["end"] += c["start"]
                chunk_segs.append(x)
            try:
                os.remove(c["path"])
            except Exception:
                pass

        # Keep whichever recovered more distinct content; report the voted language.
        if _loop_score(chunk_segs) > _loop_score(segs):
            return chunk_segs, winner
        return segs, dlang

    # 1. Split stereo into two mono 16kHz channels.
    ch_dir = Path(base_temp_dir) / "ch"
    ch_dir.mkdir(parents=True, exist_ok=True)
    left = str(ch_dir / "L.wav")
    right = str(ch_dir / "R.wav")
    _sp.run([
        'ffmpeg', '-y', '-i', str(local_audio_path),
        '-filter_complex', '[0:a]channelsplit=channel_layout=stereo[L][R]',
        '-map', '[L]', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', left,
        '-map', '[R]', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', right,
    ], check=True, capture_output=True)

    # 2. Transcribe each channel independently. Channel == speaker.
    log.info("Transcribing channel L (Speaker A)...")
    l_segs, l_lang = _transcribe_file(left, language)
    log.info("Transcribing channel R (Speaker B)...")
    r_segs, r_lang = _transcribe_file(right, language)

    for _p in (left, right):
        try:
            os.remove(_p)
        except Exception:
            pass

    # Language: trust the channel with more transcribed text (a near-silent
    # channel can mis-detect). Explicit arg always wins.
    if language:
        detected_language = language
    else:
        l_chars = sum(len(s["text"]) for s in l_segs)
        r_chars = sum(len(s["text"]) for s in r_segs)
        detected_language = l_lang if l_chars >= r_chars else r_lang

    # 3. Tag speaker by channel and combine.
    merged = []
    for s in l_segs:
        s["speaker"] = "Speaker A"
        merged.append(s)
    for s in r_segs:
        s["speaker"] = "Speaker B"
        merged.append(s)
    merged.sort(key=lambda s: s["start"])

    # 4. Collapse intra-segment loop spam, then drop consecutive identical
    # segments per speaker — both are hallucination artifacts of the server's
    # condition_on_previous_text=True.
    for s in merged:
        s["text"] = _collapse_repeats(s["text"])
    deduped = []
    for s in merged:
        if not s["text"]:
            continue
        if (deduped and deduped[-1]["speaker"] == s["speaker"]
                and deduped[-1]["text"] == s["text"]):
            deduped[-1]["end"] = s["end"]
            continue
        deduped.append(s)

    # 5. Merge consecutive same-speaker segments with no real pause into
    # readable bubbles (zero data loss).
    temp_segments = []
    for s in deduped:
        if temp_segments:
            prev = temp_segments[-1]
            if (prev["speaker"] == s["speaker"]
                    and s["start"] - prev["end"] <= 1.0
                    and (prev["end"] - prev["start"]) < 15
                    and len(prev["text"]) < 240):
                prev["text"] = (prev["text"] + " " + s["text"]).strip()
                prev["end"] = s["end"]
                continue
        temp_segments.append(dict(s))

    log.info(f"Per-channel transcription: L={len(l_segs)} R={len(r_segs)} "
             f"-> {len(temp_segments)} bubbles. Lang: {detected_language}")
    return temp_segments, detected_language


def process_audio(blob_filename: str, client_code: str, language: str = None) -> Dict[str, Any]:
    """
    Process audio file using OpenAI gpt-4o-transcribe-diarize for high performance.
    """
    with _PROCESSING_LOCK:
        try:
            gc.collect()
            log.info(f"Starting cloud-based processing for {client_code}/{blob_filename}")
            
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            base_temp_dir = Path(f"/tmp/vaidikai/{client_code}")
            base_temp_dir.mkdir(parents=True, exist_ok=True)
            transcripts_dir = base_temp_dir / "transcripts"
            outputs_dir = base_temp_dir / "outputs"
            transcripts_dir.mkdir(exist_ok=True)
            outputs_dir.mkdir(exist_ok=True)
            
            # STEP 1: Download with Fuzzy Matching
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            containers = ["client-intake", "processing"]
            full_blob_path = None
            found_container = None
            
            for container_name in containers:
                try:
                    container_client = blob_service_client.get_container_client(container_name)
                    prefix = f"{client_code}/"
                    # List blobs in client folder and find the best match
                    blobs = list(container_client.list_blobs(name_starts_with=prefix))
                    matching_blobs = [b.name for b in blobs if blob_filename.split("/")[-1] in b.name]
                    
                    if matching_blobs:
                        full_blob_path = sorted(matching_blobs)[-1] # Take the latest matching version
                        found_container = container_name
                        break
                except Exception as ce:
                    log.warning(f"Warning: Could not search container {container_name}: {ce}")
            
            if not full_blob_path:
                raise ValueError(f"No matching blob found for {blob_filename} in {client_code}/ (Searched {containers})")
            
            log.info(f"Matched blob: {full_blob_path} in container: {found_container}")
            
            pure_filename = full_blob_path.split("/")[-1]
            local_audio_path = base_temp_dir / pure_filename
            
            blob_client = blob_service_client.get_blob_client(container=found_container, blob=full_blob_path)
            with open(local_audio_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            # Detect channel layout. Dual-channel call recordings keep the two
            # speakers on separate channels -> channel == speaker (perfect
            # diarization, no cross-talk). Far better than diarizing mixed mono,
            # so when stereo we skip pyannote entirely.
            stereo = False
            try:
                import subprocess as _pf
                _pc = _pf.run([
                    'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                    '-show_entries', 'stream=channels', '-of',
                    'default=noprint_wrappers=1:nokey=1', str(local_audio_path)
                ], check=True, capture_output=True, text=True)
                stereo = int((_pc.stdout.strip() or '1')) >= 2
            except Exception as _pe:
                log.error(f"Channel probe failed: {_pe}; assuming mono.")
            log.info(f"Audio channels: {'STEREO (channel=speaker)' if stereo else 'MONO (needs diarization)'}")

            # STEP 1.5: Accurate Diarization with pyannote.audio (MONO only)
            use_pyannote = False
            speaker_segments = []
            hf_token = (os.getenv("HF_TOKEN") or os.getenv("HF_Read_token", "")).strip('"').strip("'")

            if hf_token and not stereo:
                log.info("Starting Pyannote Diarization...")
                try:
                    # Pre-validate token access to gated model before loading heavy pipeline
                    import requests as _hf_req
                    _PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
                    _config_url = f"https://huggingface.co/{_PYANNOTE_MODEL}/resolve/main/config.yaml"
                    _r = _hf_req.head(_config_url, headers={"Authorization": f"Bearer {hf_token}"}, timeout=10)
                    if _r.status_code == 401:
                        raise PermissionError(
                            "HF_TOKEN is invalid or expired. Generate a new READ token at "
                            "https://huggingface.co/settings/tokens"
                        )
                    if _r.status_code == 403:
                        raise PermissionError(
                            f"HF account has NOT accepted terms for {_PYANNOTE_MODEL}. "
                            f"Visit https://huggingface.co/{_PYANNOTE_MODEL} and click 'Agree and access repository'."
                        )
                    if _r.status_code not in (200, 302):
                        raise PermissionError(f"Unexpected HF response {_r.status_code} for {_PYANNOTE_MODEL}")
                    log.info(f"HF token validated. Access to {_PYANNOTE_MODEL} confirmed.")

                    from pyannote.audio import Pipeline
                    pipeline = Pipeline.from_pretrained(
                        _PYANNOTE_MODEL,
                        token=hf_token
                    )
                    import torch
                    if torch.cuda.is_available():
                        pipeline.to(torch.device("cuda"))
                        log.info("Using CUDA for Pyannote")
                    else:
                        pipeline.to(torch.device("cpu"))
                    import subprocess
                    temp_wav_path = str(local_audio_path) + ".16k.wav"
                    try:
                        log.info("Converting audio to 16kHz mono PCM WAV for Pyannote...")
                        subprocess.run([
                            'ffmpeg', '-y', '-i', str(local_audio_path),
                            '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                            temp_wav_path
                        ], check=True, capture_output=True)
                        # Load as an in-memory waveform and hand pyannote the
                        # tensor directly. Passing a file path makes pyannote
                        # trust the WAV header duration, but decoders often yield
                        # a few samples short (e.g. 78895 vs 80000) and pyannote
                        # hard-fails the chunk assertion. With an in-memory
                        # tensor pyannote uses the actual decoded sample count.
                        import torchaudio
                        waveform, sr = torchaudio.load(temp_wav_path)
                        log.info(f"Running Pyannote on waveform ({waveform.shape[1]} samples @ {sr}Hz)...")
                        diarization = pipeline({"waveform": waveform, "sample_rate": sr})
                    finally:
                        if os.path.exists(temp_wav_path):
                            os.remove(temp_wav_path)
                    
                    if hasattr(diarization, 'itertracks'):
                        tracks = diarization.itertracks(yield_label=True)
                    elif hasattr(diarization, 'tracks'):
                        # fallback for alternative output objects
                        tracks = diarization.tracks()
                    else:
                        log.error(f"Unknown diarization object type: {type(diarization)} - Attributes: {dir(diarization)}")
                        raise AttributeError(f"Diarization output missing itertracks: {type(diarization)}")

                    for turn, _, speaker in tracks:
                        speaker_segments.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    log.info(f"Diarization complete. Found {len(speaker_segments)} turns.")
                    use_pyannote = True
                except Exception as e:
                    log.error(f"Pyannote Diarization failed: {e}. Falling back to gap-based diarization.")
            elif stereo:
                log.info("Stereo audio: channel-based speaker separation (pyannote skipped).")
            else:
                log.info("HF_TOKEN not found. Skipping Pyannote Diarization. Falling back to gap-based.")
            
            # STEP 2: Transcribe with Local Whisper Large-v3 for maximum accuracy
            try:
                log.info("Using Groq API for lightning-fast Whisper Large-v3...")
                from openai import OpenAI
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise Exception("GROQ_API_KEY not found in environment")
                    
                groq_client = OpenAI(
                    api_key=groq_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                
                if stereo:
                    # Dual-channel: channel == speaker. No pyannote, no
                    # cross-talk -> perfect diarization + more complete text.
                    temp_segments, detected_language = transcribe_dual_channel(
                        groq_client, local_audio_path, base_temp_dir, client_code, language
                    )
                    if not temp_segments:
                        raise Exception("Dual-channel transcription produced no segments")
                else:
                    log.info("Starting Mono transcription...")
                    
                    prompt = CLIENT_PROMPT_CONFIG.get(client_code, CLIENT_PROMPT_CONFIG['DEFAULT'])
                    detected_language = language or "auto"
                    norm_segments = []
                    response_text = ""
                    
                    runpod_api_key = os.getenv("RUNPOD_API_KEY")
                    raw_url = os.getenv("RAW_RUNPOD_URL")
                    runpod_success = False
                    
                    if raw_url:
                        log.info("Trying Raw RunPod for Mono as primary...")
                        import requests
                        try:
                            with open(local_audio_path, "rb") as f:
                                files = {"file": f}
                                # See note in transcribe_dual_channel(): server
                                # honors only a fixed field set; condition_on_previous_text
                                # is dropped (fix via server redeploy). No prompt/
                                # hotwords — domain bias in the wrong language seeds
                                # hallucination loops; auto-detect instead.
                                data = {
                                    "model": "Systran/faster-whisper-large-v3",
                                    "response_format": "verbose_json",
                                    "timestamp_granularities": "segment",
                                    "temperature": "0.0",
                                    "vad_filter": "true",
                                }
                                if language:
                                    data["language"] = language
                                
                                resp = requests.post(raw_url, files=files, data=data, timeout=600)
                                resp.raise_for_status()
                                r_json = resp.json()
                            
                            detected_language = r_json.get("language", detected_language)
                            response_text = r_json.get("text", "")
                            
                            for s in (r_json.get("segments", []) or []):
                                st = s.get("start", 0.0) or 0.0
                                en = s.get("end", 0.0) or 0.0
                                txt = (s.get("text", "") or "").strip()
                                lp = s.get("avg_logprob", 0.0) or 0.0
                                norm_segments.append({
                                    "text": txt, "start": st, "end": en, "avg_logprob": lp
                                })
                            runpod_success = True
                            log.info("RunPod Mono transcription successful.")
                        except Exception as e:
                            log.error(f"RunPod Mono failed: {e}. Falling back to Groq...")
                            runpod_success = False
                            
                    if not runpod_success:
                        log.info("Starting transcription with Groq (Whisper Large-v3) fallback...")
                        _mkw = dict(
                            model="whisper-large-v3",
                            response_format="verbose_json",
                            timestamp_granularities=["segment"],
                            temperature=0,
                            prompt=prompt
                        )
                        if language:
                            _mkw["language"] = language
                        with open(local_audio_path, "rb") as f:
                            response = groq_client.audio.transcriptions.create(file=f, **_mkw)
    
                        detected_language = getattr(response, 'language', None) or detected_language
                        raw_segments = getattr(response, 'segments', []) or []
                        response_text = getattr(response, 'text', '') or ''
                        
                        norm_segments = []
                        for seg in raw_segments:
                            is_d = isinstance(seg, dict)
                            norm_segments.append({
                                "text": ((seg.get('text') if is_d else getattr(seg, 'text', '')) or ''),
                                "start": ((seg.get('start') if is_d else getattr(seg, 'start', 0)) or 0),
                                "end": ((seg.get('end') if is_d else getattr(seg, 'end', 0)) or 0),
                                "avg_logprob": ((seg.get('avg_logprob') if is_d else getattr(seg, 'avg_logprob', 0.0)) or 0.0),
                            })
                            
                    # Last resort: no segments but we have text -> single block.
                    if not norm_segments and response_text.strip():
                        norm_segments = [{"text": response_text.strip(), "start": 0, "end": 0, "avg_logprob": 0.0}]

                    log.info(f"Segments parsed: {len(norm_segments)} | full text length: {len(response_text)} chars")

                    temp_segments = []
                    # Default to Speaker B (Customer) for mono files
                    current_speaker = "Speaker B"
                    last_end_time = 0
                    for seg in norm_segments:
                        text = (seg["text"] or "").strip()
                        start = seg["start"]
                        end = seg["end"]
                        avg_logprob = seg["avg_logprob"]
                        if avg_logprob is None:
                            avg_logprob = 0.0

                        if use_pyannote and speaker_segments:
                            dominant_speaker = "Unknown"
                            max_overlap = 0
                            for p_seg in speaker_segments:
                                overlap = min(end, p_seg["end"]) - max(start, p_seg["start"])
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    dominant_speaker = p_seg["speaker"]
                            if dominant_speaker != "Unknown":
                                current_speaker = dominant_speaker
                        else:
                            # Gap-based fallback: assume mono is a single speaker if Pyannote is disabled
                            pass

                        if filter_segment(text, avg_logprob):
                            if start == 0 and end == 0 and temp_segments:
                                last_end_time = end
                                continue
                            norm = text.strip()
                            recent = [seg2["text"].strip() for seg2 in temp_segments[-2:]]
                            if len(recent) == 2 and recent[0] == norm and recent[1] == norm:
                                # 3rd+ identical line in a row = Whisper
                                # hallucination loop. Single/double repeats
                                # ("haan ji", "ok") are legit — keep them.
                                last_end_time = end
                                continue
                            temp_segments.append({
                                "start": start, "end": end, "text": text,
                                "speaker": current_speaker, "avg_logprob": avg_logprob
                            })
                        last_end_time = end

            except Exception as e:
                log.error(f"Local Whisper failed: {e}. Falling back to OpenAI whisper-1 API...")
                temp_segments = []
                max_retries = 3
                response = None
                for attempt in range(max_retries):
                    try:
                        with open(local_audio_path, "rb") as f:
                            _wkwargs = dict(
                                file=f,
                                model="whisper-1",
                                response_format="verbose_json",
                                timestamp_granularities=["segment"],
                                prompt=CLIENT_PROMPT_CONFIG.get(client_code, CLIENT_PROMPT_CONFIG['DEFAULT'])
                            )
                            if language:
                                _wkwargs["language"] = language
                            response = client.audio.transcriptions.create(**_wkwargs)
                        break
                    except Exception as api_e:
                        log.error(f"API fallback attempt {attempt+1} failed: {api_e}")
                        if attempt < max_retries - 1:
                            time.sleep(3 * (attempt + 1))
                        else:
                            raise Exception(f"Both local and API transcription failed: {api_e}")

                detected_language = getattr(response, 'language', None) or 'auto'
                raw_segments = getattr(response, 'segments', [])
                current_speaker = "Speaker A"
                last_end_time = 0
                for s in raw_segments:
                    is_dict = isinstance(s, dict)
                    text = s.get('text', '').strip() if is_dict else s.text.strip()
                    start = s.get('start', 0) if is_dict else s.start
                    end = s.get('end', 0) if is_dict else s.end
                    avg_logprob = s.get('avg_logprob', 0.0) if is_dict else getattr(s, 'avg_logprob', 0.0)
                    if avg_logprob is None:
                        avg_logprob = 0.0

                    if use_pyannote and speaker_segments:
                        dominant_speaker = "Unknown"
                        max_overlap = 0
                        for p_seg in speaker_segments:
                            overlap = min(end, p_seg["end"]) - max(start, p_seg["start"])
                            if overlap > max_overlap:
                                max_overlap = overlap
                                dominant_speaker = p_seg["speaker"]
                        if dominant_speaker != "Unknown":
                            current_speaker = dominant_speaker
                    else:
                        # Gap-based fallback: assume mono is a single speaker if Pyannote is disabled
                        pass

                    if filter_segment(text, avg_logprob):
                        if start == 0 and end == 0 and temp_segments:
                            continue
                        if temp_segments and temp_segments[-1]["text"].strip() == text.strip():
                            continue
                        temp_segments.append({
                            "start": start, "end": end, "text": text,
                            "speaker": current_speaker, "avg_logprob": avg_logprob
                        })
                    last_end_time = end

            # AI Speaker Role Mapping
            role_mapping = identify_speaker_roles(temp_segments, client_code)
            
            # STEP 5: Final Enrichment and Role Mapping
            processed_segments = []
            final_raw_segments = []
            
            for i, s in enumerate(temp_segments):
                speaker_id = s['speaker']
                role = role_mapping.get(speaker_id, speaker_id)
                text = s['text']
                confidence = round((1 + s['avg_logprob']) * 100, 1)
                confidence = max(0, min(100, confidence))
                
                segment_data = {
                    'segment_id': f"SEG-{i+1:03d}",
                    'language': detected_language,
                    'start_time': round(s['start'], 2),
                    'end_time': round(s['end'], 2),
                    'speaker': role,
                    'transcript': text,
                    'intent': tag(text, INTENT_RULES),
                    'sentiment': tag(text, SENTIMENT_RULES),
                    'outcome': tag(text, OUTCOME_RULES),
                    'key_signal': get_key_signal(text),
                    'confidence': confidence,
                    'qa_status': calculate_qa_status(confidence),
                    'notes': ''
                }
                processed_segments.append(segment_data)
                final_raw_segments.append({**s, "speaker": role})

            # STEP 6: Save Results
            transcript_filename = f"{pure_filename}_transcript.json"

            transcript_path = transcripts_dir / transcript_filename
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump({"language": detected_language, "segments": final_raw_segments}, f, indent=2, ensure_ascii=False)
            
            processed_filename = f"{pure_filename}_processed.json"

            processed_path = outputs_dir / processed_filename
            with open(processed_path, 'w', encoding='utf-8') as f:
                json.dump({"segments": processed_segments, "language": detected_language}, f, indent=2, ensure_ascii=False)
            
            # STEP 7: Upload transcript + processed JSON to Azure, cleanup local audio
            for blob_name, local_path in [
                (f"{client_code}/{transcript_filename}", transcript_path),
                (f"{client_code}/{processed_filename}", processed_path),
            ]:
                bc = blob_service_client.get_blob_client(container="processing", blob=blob_name)
                with open(local_path, 'rb') as f:
                    bc.upload_blob(f, overwrite=True)

            if os.path.exists(local_audio_path):
                os.remove(local_audio_path)

            high_conf = sum(1 for s in processed_segments if s['confidence'] >= 65)
            review_req = sum(1 for s in processed_segments if 40 <= s['confidence'] < 65)

            processed_blob = f"{client_code}/{processed_filename}"

            return {
                "status": "success", "segments": len(processed_segments),
                "high_confidence": high_conf, "review_needed": review_req,
                "processed_file": str(processed_path),
                "processed_blob": processed_blob,
                "client_code": client_code,
                "original_filename": pure_filename, "engine": "gpt-4o", "language": detected_language
            }

        except Exception as e:
            error_msg = f"Error in processor: {str(e)}"
            log.error(error_msg)
            return {"status": "error", "error": error_msg}
