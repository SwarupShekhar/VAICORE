import os
import sys

# Ensure we're in the right directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .database import SessionLocal, engine
from . import models
from . import auth

# Ensure tables exist
models.Base.metadata.create_all(bind=engine)

def create_admin():
    db = SessionLocal()
    username = "admin"
    password = "password123"
    
    user = db.query(models.User).filter(models.User.username == username).first()
    if user:
        print(f"User '{username}' already exists. You can log in with your password.")
    else:
        hashed_pw = auth.get_password_hash(password)
        new_user = models.User(username=username, hashed_password=hashed_pw)
        db.add(new_user)
        db.commit()
        print(f"✅ Successfully created isolated user!")
        print(f"Username: {username}")
        print(f"Password: {password}")
    
    db.close()

if __name__ == "__main__":
    create_admin()
