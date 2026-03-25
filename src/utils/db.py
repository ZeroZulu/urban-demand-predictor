"""
Database connection helpers.
Uses environment variable DATABASE_URL or falls back to .env file.
"""
import os
from functools import lru_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine."""
    url = os.getenv(
        "DATABASE_URL",
        "postgresql://analyst:localdev@localhost:5432/urban_demand",
    )
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def test_connection() -> bool:
    """Return True if DB is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
