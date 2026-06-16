import runpod
import stable_whisper
import requests
import os
import uuid

# Load model globally during cold start
print("Loading Whisper Large-V3 model...")
model = stable_whisper.load_model('large-v3')
print("Model loaded successfully!")

def handler(job):
    job_input = job['input']
    audio_url = job_input.get('audio_url')
    do_diarization = job_input.get('do_diarization', False)
    hf_token = job_input.get('hf_token', None)

    if not audio_url:
        return {"error": "Missing audio_url in input"}

    # Download the audio file
    temp_id = str(uuid.uuid4())
    audio_path = f"/tmp/{temp_id}_audio.mp3"
    
    print(f"Downloading audio from {audio_url}...")
    try:
        r = requests.get(audio_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(audio_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        return {"error": f"Failed to download audio: {str(e)}"}

    print("Transcribing...")
    try:
        if do_diarization and hf_token:
            print("Diarization enabled. Running Pyannote alongside Whisper...")
            result = model.transcribe(audio_path, hf_token=hf_token)
        else:
            print("Running standard Whisper without diarization...")
            result = model.transcribe(audio_path)
            
        print("Applying 42-character line limits...")
        result.split_by_length(max_chars=42)
        
        srt_path = f"/tmp/{temp_id}_output.srt"
        result.to_srt_vtt(srt_path, word_level=False)
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
            
        # Cleanup
        if os.path.exists(audio_path): os.remove(audio_path)
        if os.path.exists(srt_path): os.remove(srt_path)
        
        print("Transcription complete!")
        return {"srt": srt_content}
        
    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}

runpod.serverless.start({"handler": handler})
