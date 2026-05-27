import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool, create_engine
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from dotenv import load_dotenv

# Import Base from models.py
from models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Load env variables from .env if present
load_dotenv()

# Read DATABASE_URL from os.environ
database_url = os.environ.get("DATABASE_URL")

# Check if DATABASE_URL is set and valid, otherwise raise a clear exception
if not database_url or database_url.strip() == "DATABASE_URL":
    raise ValueError(
        "DATABASE_URL environment variable is not configured. "
        "Please ensure DATABASE_URL is set in your environment or defined in your .env file."
    )

# Override the sqlalchemy.url configuration option dynamically
config.set_main_option("sqlalchemy.url", database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target_metadata
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = database_url
    
    # Strip async driver prefix if present for offline compilation compatibility
    if url and "+asyncpg" in url:
        url = url.replace("+asyncpg", "")
        
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_async() -> None:
    """Run migrations using an AsyncEngine."""
    url = database_url
    
    # Create the async engine
    connectable = create_async_engine(
        url,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_sync() -> None:
    """Run migrations using a standard SyncEngine."""
    url = database_url
    
    # Strip asyncpg prefix if we accidentally pass an async URL to a sync engine
    if url and "+asyncpg" in url:
        url = url.replace("+asyncpg", "")
        
    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    url = database_url
    
    if url and ("+asyncpg" in url or "+aiopg" in url):
        # Support async-aware setup
        asyncio.run(run_migrations_async())
    else:
        # Support sync setup
        run_migrations_sync()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
