import asyncio
from sqlalchemy import text
from database import get_db_session

async def clean_dashboard():
    print("Connecting to database...")
    async with get_db_session() as session:
        # Delete all records from the upload_logs table
        await session.execute(text("DELETE FROM upload_logs;"))
        await session.commit()
        print("Successfully wiped all old test data from the dashboard!")

if __name__ == "__main__":
    asyncio.run(clean_dashboard())
