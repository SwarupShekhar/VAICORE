from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    
    jobs = relationship("TranscriptionJob", back_populates="owner")

class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"

    id = Column(String, primary_key=True, index=True) # UUID string
    filename = Column(String)
    status = Column(String, default="Pending") # Pending, Processing, Completed, Failed
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    srt_content = Column(String, nullable=True) # Storing SRT directly or file path
    owner_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="jobs")
