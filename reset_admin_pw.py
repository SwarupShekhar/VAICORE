import asyncio
from database import get_db_session
from models import User, hash_password
from sqlalchemy import select
import os

async def reset_pw():
    async with get_db_session() as db:
        email = os.getenv("SUPER_ADMIN_EMAIL", "admin@vaidik.ai").lower().strip()
        user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user:
            pw = os.getenv("ADMIN_PASSWORD", "vaicore-admin-2026")
            user.hashed_password = hash_password(pw)
            await db.commit()
            print(f"Password for {email} reset successfully.")
        else:
            print("Admin user not found in database.")

if __name__ == "__main__":
    asyncio.run(reset_pw())
