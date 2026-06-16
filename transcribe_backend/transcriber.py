import os
import datetime
import traceback
from .database import SessionLocal, engine
from . import models

# Lazy load model to prevent PyTorch multiprocessing deadlocks on macOS when RQ forks
model = None

def process_transcription_job(job_id: str, file_path: str):
    # Move import inside the function so it only imports AFTER rq forks!
    import stable_whisper
    
    # Fix SQLite multi-threading deadlock across forked process boundaries
    engine.dispose()
    db = SessionLocal()
    job = db.query(models.TranscriptionJob).filter(models.TranscriptionJob.id == job_id).first()
    
    if not job:
        db.close()
        return

    try:
        # Mark as processing
        job.status = "Processing"
        db.commit()
        
        print(f"[Job {job_id}] Starting transcription for {file_path}")
        
        global model
        if model is None:
            print("Loading Whisper model into memory...")
            model = stable_whisper.load_model('base')

        # Callback to update progress in DB
        def progress_callback(seek, total):
            if total > 0:
                # Calculate percentage
                progress_pct = round((seek / total) * 100, 2)
                # Ensure it doesn't exceed 100%
                progress_pct = min(100.0, progress_pct)
                
                # Fetch fresh job to avoid stale data issues
                current_job = db.query(models.TranscriptionJob).filter(models.TranscriptionJob.id == job_id).first()
                if current_job:
                    current_job.progress = progress_pct
                    db.commit()

        # Run Whisper via stable-ts
        # This wrapper prevents hallucinations and allows strict SRT formatting
        result = model.transcribe(file_path, language=None, progress_callback=progress_callback) # language=None allows auto-detect, or specify if needed
        
        # Save output as SRT adhering to TV broadcast constraints (Max 42 chars per line)
        srt_path = file_path.rsplit('.', 1)[0] + '.srt'
        result.split_by_length(max_chars=42)
        result.to_srt_vtt(srt_path, word_level=False)
        
        # Update job success
        job.status = "Completed"
        job.completed_at = datetime.datetime.utcnow()
        # Optionally, read the srt content to store in DB, but storing path is safer for large files
        with open(srt_path, "r", encoding="utf-8") as f:
            job.srt_content = f.read()
            
        db.commit()
        print(f"[Job {job_id}] Transcription complete. Saved to {srt_path}")
        
    except Exception as e:
        print(f"[Job {job_id}] Failed. Error: {traceback.format_exc()}")
        db.rollback() # Rollback the session
        job.status = "Failed"
        db.commit()
    finally:
        db.close()
