"""Database engine, session factory and lifecycle helpers.

The SQLite file lives inside this project's data/ directory and is never
shared with the other MVP project.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from backend.config import Settings, get_settings
from backend.models import Base


def make_database_url(settings: Settings) -> str:
    """Resolve a relative sqlite:/// path against the project root."""
    url = settings.database_url
    if url.startswith("sqlite:///"):
        raw = url[len("sqlite:///") :]
        path = Path(raw)
        if not path.is_absolute():
            path = Path(settings.__class__.PROJECT_ROOT if hasattr(settings.__class__, "PROJECT_ROOT") else Path(__file__).resolve().parent.parent) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
    return url


def create_engine_and_session(url: str) -> tuple[object, sessionmaker]:
    connect_args: dict = {}
    # check_same_thread is SQLite-only; PostgreSQL drivers reject it.
    if url.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
    engine = create_engine(
        url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
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
