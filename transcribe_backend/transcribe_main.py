from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import uuid
import datetime
import os

from database import engine, get_db
import models
import auth
from redis import Redis
from rq import Queue
from transcriber import process_transcription_job

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Samsung Transcription Portal API")

# Setup Redis Queue
redis_conn = Redis()
q = Queue(connection=redis_conn)

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

# --- AUTHENTICATION ROUTES ---

@app.post("/api/register")
def register_user(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth.get_password_hash(form_data.password)
    new_user = models.User(username=form_data.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User registered successfully"}

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Read the master password directly from Vaicore's .env file
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path)
    
    admin_password = os.getenv("ADMIN_PASSWORD")
    
    # Check if the user is using the Vaicore admin credentials
    if form_data.username == "admin" and form_data.password == admin_password:
        # Ensure the admin user exists in the local SQLite DB for job tracking
        user = db.query(models.User).filter(models.User.username == "admin").first()
        if not user:
            user = models.User(username="admin", hashed_password="synced_with_vaicore")
            db.add(user)
            db.commit()
            
        access_token_expires = datetime.timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth.create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password. Use your Vaicore admin credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- JOB ROUTES ---

@app.post("/api/jobs/upload")
async def upload_audio(
    file: UploadFile = File(...), 
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    # Save the file (Synchronous for now, will handle chunked later if needed)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Create job record
    new_job = models.TranscriptionJob(
        id=job_id,
        filename=file.filename,
        status="Queued",
        owner_id=current_user.id
    )
    db.add(new_job)
    db.commit()
    
    # Push to Redis Queue with 1 hour timeout
    q.enqueue(process_transcription_job, job_id, file_path, job_timeout='1h')
    
    return {"job_id": job_id, "status": "Queued", "filename": file.filename}

from pydantic import BaseModel

class ChunkCompleteRequest(BaseModel):
    filename: str

@app.post("/api/jobs/upload/chunk")
async def upload_chunk(
    job_id: str,
    chunk_index: int,
    total_chunks: int,
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user)
):
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{job_id}")
    
    # Append the chunk to the temp file
    with open(temp_path, "ab") as f:
        content = await file.read()
        f.write(content)
        
    return {"status": "Chunk received", "chunk_index": chunk_index}

@app.post("/api/jobs/upload/complete/{job_id}")
def complete_chunk_upload(
    job_id: str,
    request: ChunkCompleteRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db)
):
    temp_path = os.path.join(UPLOAD_DIR, f"temp_{job_id}")
    final_path = os.path.join(UPLOAD_DIR, f"{job_id}_{request.filename}")
    
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=400, detail="Temporary file not found")
        
    os.rename(temp_path, final_path)
    
    # Create job record
    new_job = models.TranscriptionJob(
        id=job_id,
        filename=request.filename,
        status="Queued",
        owner_id=current_user.id
    )
    db.add(new_job)
    db.commit()
    
    # Push to Redis Queue with long timeout
    q.enqueue(process_transcription_job, job_id, final_path, job_timeout='4h')
    
    return {"job_id": job_id, "status": "Queued", "filename": request.filename}

@app.get("/api/jobs")
def get_jobs(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    jobs = db.query(models.TranscriptionJob).filter(models.TranscriptionJob.owner_id == current_user.id).all()
    return jobs

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    job = db.query(models.TranscriptionJob).filter(models.TranscriptionJob.id == job_id, models.TranscriptionJob.owner_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

from fastapi.responses import FileResponse

@app.get("/api/jobs/{job_id}/download")
def download_job_srt(job_id: str, current_user: models.User = Depends(auth.get_current_user_with_query), db: Session = Depends(get_db)):
    job = db.query(models.TranscriptionJob).filter(models.TranscriptionJob.id == job_id, models.TranscriptionJob.owner_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "Completed":
        raise HTTPException(status_code=400, detail="Transcription is not completed yet")
        
    srt_filename = f"{job_id}_{job.filename.rsplit('.', 1)[0]}.srt"
    file_path = os.path.join(UPLOAD_DIR, srt_filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="SRT file not found on server")
        
    return FileResponse(path=file_path, filename=f"{job.filename.rsplit('.', 1)[0]}.srt", media_type="application/x-subrip")

@app.get("/api/jobs/{job_id}/media")
def get_job_media(job_id: str, current_user: models.User = Depends(auth.get_current_user_with_query), db: Session = Depends(get_db)):
    job = db.query(models.TranscriptionJob).filter(models.TranscriptionJob.id == job_id, models.TranscriptionJob.owner_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{job.filename}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Media file not found on server")
        
    return FileResponse(path=file_path)

@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    job = db.query(models.TranscriptionJob).filter(models.TranscriptionJob.id == job_id, models.TranscriptionJob.owner_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Remove files from disk
    media_path = os.path.join(UPLOAD_DIR, f"{job_id}_{job.filename}")
    srt_path = os.path.join(UPLOAD_DIR, f"{job_id}_{job.filename.rsplit('.', 1)[0]}.srt")
    
    if os.path.exists(media_path):
        os.remove(media_path)
    if os.path.exists(srt_path):
        os.remove(srt_path)
        
    # Delete from DB
    db.delete(job)
    db.commit()
    return {"message": "Job deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
