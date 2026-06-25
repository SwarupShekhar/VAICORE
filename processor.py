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
    'CLIENT002': "This is a conversation in an Indian language mixed with English. Transcribe exactly as spoken in the native script. Do not translate. Do not output any Chinese or foreign characters. If the language is Hindi, strictly use Devanagari script, NEVER use Urdu script. Haan ji, okay.",
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
    """Dual-channel call: transcribe the MIXED audio once, diarize by channel energy.

    Transcribing each channel in isolation hallucinates badly: an isolated
    channel is silent while the other party talks, and Whisper invents text +
    guesses a random language (Tamil/Telugu on a Hindi call) on silence.

    Instead:
      1. Transcribe the full MIXED audio once -> full signal, stable language
         auto-detect, no silence hallucination, best accuracy.
      2. Split L/R only to MEASURE energy. For each transcribed segment, the
         louder channel over that window is the speaker. Deterministic, no
         pyannote, no guessing.
    Returns (temp_segments, detected_language).
    """
    import subprocess as _sp
    import wave as _wave
    import array as _array
    import math as _math

    # 1. Transcribe the full mixed audio once.
    prompt = CLIENT_PROMPT_CONFIG.get(client_code, CLIENT_PROMPT_CONFIG['DEFAULT'])
    detected_language = language or "auto"
    raw = []
    
    # Primary: RunPod Serverless (requests.post bypass for timestamp_granularities)
    runpod_api_key = os.getenv("RUNPOD_API_KEY")
    runpod_endpoint = os.getenv("RUNPOD_WHISPER_ENDPOINT")
    runpod_success = False
    
    if runpod_api_key and runpod_endpoint:
        log.info("Trying RunPod as primary transcription engine...")
        import requests
        url = f"https://api.runpod.ai/v2/{runpod_endpoint}/runsync"
        headers = {
            "Authorization": f"Bearer {runpod_api_key}",
            "Content-Type": "application/json"
        }
        
        import base64
        runpod_mp3 = str(local_audio_path) + ".runpod.mp3"
        subprocess.run(["ffmpeg", "-y", "-i", local_audio_path, "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k", runpod_mp3], check=True, capture_output=True)
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
        try:
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
                
            r_json = runpod_data.get("output", {})
            
            detected_language = r_json.get("language", detected_language)
            # Parse segments
            for s in (r_json.get("segments", []) or []):
                st = s.get("start", 0.0) or 0.0
                en = s.get("end", 0.0) or 0.0
                txt = (s.get("text", "") or "").strip()
                lp = s.get("avg_logprob", 0.0) or 0.0
                raw.append({"start": st, "end": en, "text": txt, "avg_logprob": lp})
            runpod_success = True
            log.info("RunPod transcription successful.")
        except Exception as e:
            log.error(f"RunPod failed: {e}. Falling back to Groq...")
            runpod_success = False

    # Fallback: Groq SDK
    if not runpod_success:
        log.info("Using Groq API fallback...")
        _kw = dict(
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            temperature=0,  # anti-hallucination
            prompt=prompt,
        )
        if language:
            _kw["language"] = language
        
        with open(local_audio_path, "rb") as f:
            r = groq_client.audio.transcriptions.create(file=f, **_kw)
        
        detected_language = getattr(r, 'language', None) or detected_language
        raw_segs = getattr(r, 'segments', []) or []
        for s in raw_segs:
            isd = isinstance(s, dict)
            st = (s.get("start") if isd else getattr(s, "start", 0.0)) or 0.0
            en = (s.get("end") if isd else getattr(s, "end", 0.0)) or 0.0
            txt = (s.get("text") if isd else getattr(s, "text", "")).strip()
            lp = (s.get("avg_logprob") if isd else getattr(s, "avg_logprob", 0.0)) or 0.0
            raw.append({"start": st, "end": en, "text": txt, "avg_logprob": lp})

    log.info(f"Mixed transcription: {len(raw)} segments. Lang: {detected_language}")

    # 2. Split channels (energy measurement only — never transcribed).
    ch_dir = Path(base_temp_dir) / "ch"
    ch_dir.mkdir(exist_ok=True)
    left = str(ch_dir / "L.wav")
    right = str(ch_dir / "R.wav")
    _sp.run([
        'ffmpeg', '-y', '-i', str(local_audio_path),
        '-filter_complex', '[0:a]channelsplit=channel_layout=stereo[L][R]',
        '-map', '[L]', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', left,
        '-map', '[R]', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', right,
    ], check=True, capture_output=True)

    def _load(path):
        with _wave.open(path, 'rb') as w:
            sr = w.getframerate()
            data = w.readframes(w.getnframes())
        return _array.array('h', data), sr

    Ld, sr = _load(left)
    Rd, _ = _load(right)
    for _p in (left, right):
        try:
            os.remove(_p)
        except Exception:
            pass

    def _rms(arr, t0, t1):
        s0 = max(0, int(t0 * sr))
        s1 = min(len(arr), int(t1 * sr))
        if s1 <= s0:
            return 0.0
        step = max(1, (s1 - s0) // 4000)  # subsample long windows for speed
        acc = 0.0
        cnt = 0
        for i in range(s0, s1, step):
            v = arr[i]
            acc += v * v
            cnt += 1
        return _math.sqrt(acc / cnt) if cnt else 0.0

    # 3. Assign speaker per segment by louder channel over its time window.
    merged = []
    for seg in raw:
        isd = isinstance(seg, dict)
        st = (seg.get('start') if isd else getattr(seg, 'start', 0)) or 0
        en = (seg.get('end') if isd else getattr(seg, 'end', 0)) or 0
        txt = ((seg.get('text') if isd else getattr(seg, 'text', '')) or '').strip()
        if not txt:
            continue
        lp = (seg.get('avg_logprob') if isd else getattr(seg, 'avg_logprob', 0.0)) or 0.0
        el = _rms(Ld, st, en)
        er = _rms(Rd, st, en)
        merged.append({
            "start": st,
            "end": en,
            "text": txt,
            "speaker": "Speaker A" if el >= er else "Speaker B",
            "avg_logprob": lp,
        })

    merged.sort(key=lambda s: s["start"])

    # 4. Merge consecutive same-speaker segments when there's no real pause and
    # the bubble isn't too long (readable bubbles, zero data loss).
    temp_segments = []
    for s in merged:
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

    log.info(f"Channel-energy diarization: {len(merged)} segs -> {len(temp_segments)} bubbles.")
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
                    
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
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
                    runpod_endpoint = os.getenv("RUNPOD_WHISPER_ENDPOINT")
                    runpod_success = False
                    
                    if runpod_api_key and runpod_endpoint:
                        log.info("Trying RunPod for Mono as primary...")
                        import requests
                        url = f"https://api.runpod.ai/v2/{runpod_endpoint}/runsync"
                        headers = {
                            "Authorization": f"Bearer {runpod_api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        import base64
                        runpod_mp3 = str(local_audio_path) + ".runpod.mp3"
                        subprocess.run(["ffmpeg", "-y", "-i", local_audio_path, "-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", runpod_mp3], check=True, capture_output=True)
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
                            
                        try:
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
                                
                            r_json = runpod_data.get("output", {})
                            
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
                    current_speaker = "Speaker A"
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
                            # Gap-based fallback: 0.5s silence = speaker change
                            if start - last_end_time > 0.5:
                                current_speaker = "Speaker B" if current_speaker == "Speaker A" else "Speaker A"

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
                        # Gap-based fallback: 0.5s silence = speaker change
                        if start - last_end_time > 0.5:
                            current_speaker = "Speaker B" if current_speaker == "Speaker A" else "Speaker A"

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
