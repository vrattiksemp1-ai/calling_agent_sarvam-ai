"""Database engine, session factory and lifecycle helpers.

Production uses PostgreSQL. SQLite remains available for unit tests only.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from backend.config import Settings
from backend.models import Base


def is_sqlite_url(url: str) -> bool:
    return (url or "").startswith("sqlite")


def is_postgres_url(url: str) -> bool:
    raw = (url or "").lower()
    return raw.startswith("postgres://") or raw.startswith("postgresql")


def make_database_url(settings: Settings) -> str:
    """Normalize DATABASE_URL for SQLAlchemy (psycopg2 / relative SQLite)."""
    url = (settings.database_url or "").strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]
    if url.startswith("sqlite:///"):
        raw = url[len("sqlite:///") :]
        path = Path(raw)
        if not path.is_absolute():
            root = Path(__file__).resolve().parent.parent
            path = root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
    return url


def create_engine_and_session(url: str) -> tuple[Engine, sessionmaker]:
    connect_args: dict = {}
    kwargs: dict = {"pool_pre_ping": True}
    if is_sqlite_url(url):
        connect_args["check_same_thread"] = False
        connect_args["timeout"] = 15.0
        kwargs["connect_args"] = connect_args
    else:
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20
        kwargs["pool_recycle"] = 1800

    engine = create_engine(url, **kwargs)
    if is_sqlite_url(url):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=15000")
            cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, session_factory


@contextmanager
def session_scope(session_factory: sessionmaker) -> Iterator[OrmSession]:
    db = session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
