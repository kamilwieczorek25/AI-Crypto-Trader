"""SQLAlchemy async engine and session factory."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def create_tables() -> None:
    """Create all tables (called at startup) and apply lightweight migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Add columns introduced after initial schema (idempotent ALTERs)
        await _apply_migrations(conn)


async def _apply_migrations(conn) -> None:
    """Add missing columns to existing tables. Each ALTER is safe to re-run."""
    import logging
    log = logging.getLogger(__name__)
    migrations = [
        ("positions", "highest_price", "REAL DEFAULT 0.0"),
        ("positions", "trailing_stop_pct", "REAL DEFAULT 0.0"),
        ("claude_decisions", "risk_profile", "VARCHAR(20)"),
        ("claude_decisions", "error", "TEXT"),
        ("trades", "fee_usdt", "REAL DEFAULT 0.0"),
        ("positions", "source", "VARCHAR(10) DEFAULT 'bot'"),
        ("positions", "tp_activated", "INTEGER DEFAULT 0"),
        ("positions", "tp_peak_price", "REAL DEFAULT 0.0"),
    ]
    for table, column, col_def in migrations:
        try:
            await conn.execute(
                __import__("sqlalchemy").text(
                    f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
                )
            )
            log.info("Migration: added %s.%s", table, column)
        except Exception:
            pass  # column already exists


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session


async def load_bot_state() -> dict[str, str]:
    """Load all persisted bot state key-value pairs."""
    from app.models.bot_state import BotState
    from sqlalchemy import select
    async with AsyncSessionLocal() as session:
        rows = await session.execute(select(BotState))
        return {r.key: r.value for r in rows.scalars().all()}


async def save_bot_state(key: str, value: str) -> None:
    """Upsert a single bot state key."""
    from datetime import datetime, timezone
    from app.models.bot_state import BotState
    async with AsyncSessionLocal() as session:
        existing = await session.get(BotState, key)
        if existing:
            existing.value = value
            existing.updated_at = datetime.now(timezone.utc)
        else:
            session.add(BotState(key=key, value=value))
        await session.commit()
