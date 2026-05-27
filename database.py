import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Import Base from models.py
from models import Base

# Setup structured logging
logger = logging.getLogger("vaidikai.database")
logging.basicConfig(level=logging.INFO)

# Load env variables from .env if present
load_dotenv()

# Retrieve DATABASE_URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set!")
    raise ValueError("DATABASE_URL environment variable is not set.")

logger.info("Initializing async database engine...")

# Create Async Engine with specific pool sizing
engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Create AsyncSession Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# FastAPI route dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an async database session.
    Automatically handles rollback on exceptions and session closing.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.exception("Database transaction failed, rolling back...")
            await session.rollback()
            raise e
        finally:
            await session.close()


# Standalone context manager for background tasks
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager yielding a database session.
    Use this inside background tasks or script executions outside FastAPI routes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.exception("Database operation in background task failed, rolling back...")
            await session.rollback()
            raise e
        finally:
            await session.close()


# Database table creator (for testing only)
async def create_tables() -> None:
    """
    Creates all tables defined in models.py.
    Used for local testing only; migrations handle production schema upgrades.
    """
    logger.warning("Running database table creation (create_tables)...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("All database tables created successfully.")
    except Exception as e:
        logger.exception("Failed to create database tables!")
        raise e
