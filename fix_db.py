import asyncio
from sqlalchemy import text
from database import get_db_session

async def fix():
    print("Fixing database...")
    async with get_db_session() as session:
        # 1. Clear out hardcoded project_ids that might refer to old project 8
        await session.execute(text("UPDATE clients SET project_ids = '{}' WHERE client_code = 'CLIENT002';"))
        
        # 2. Safely add the 'vad' enum using a raw connection to avoid transaction block errors on ALTER TYPE
        try:
            await session.execute(text("ALTER TYPE job_category_enum ADD VALUE IF NOT EXISTS 'vad';"))
        except Exception as e:
            print(f"Enum might already exist or error: {e}")
            
        await session.commit()
        print("Done! Database fixed.")

if __name__ == "__main__":
    asyncio.run(fix())
