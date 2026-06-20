import asyncio
from sqlalchemy import text
from database import get_db_session

async def fix():
    print("Fixing database...")
    async with get_db_session() as session:
        # 1. Clear out hardcoded project_ids that might refer to old project 8
        await session.execute(text("UPDATE clients SET project_ids = '{}' WHERE client_code = 'CLIENT002';"))
        await session.commit()
        print("Updated CLIENT002 project_ids.")
        
    # 2. Safely add the 'vad' enum using autocommit
    from database import async_engine
    async with async_engine.connect() as conn:
        await conn.execution_options(isolation_level="AUTOCOMMIT")
        try:
            await conn.execute(text("ALTER TYPE job_category_enum ADD VALUE IF NOT EXISTS 'vad';"))
            print("Enum 'vad' added to job_category_enum.")
        except Exception as e:
            print(f"Enum might already exist or error: {e}")
            
    print("Done! Database fixed.")

if __name__ == "__main__":
    asyncio.run(fix())
