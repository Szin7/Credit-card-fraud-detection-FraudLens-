"""Engine/session management. The default engine is created lazily (on
first real use, not on import) so importing this module never requires
Postgres to be reachable — important for tests and for any code path that
imports src.database without needing the database.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base
from src.utils.config import settings

_default_engine: Engine | None = None
_default_session_factory: sessionmaker | None = None


def create_db_engine(database_url: str | None = None) -> Engine:
    engine = create_engine(database_url or settings.database_url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
    return engine


def create_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


def get_session_factory() -> sessionmaker:
    """The process-wide default session factory, targeting settings.database_url."""
    global _default_engine, _default_session_factory
    if _default_session_factory is None:
        _default_engine = create_db_engine()
        _default_session_factory = create_session_factory(_default_engine)
    return _default_session_factory


def init_db(engine: Engine | None = None) -> None:
    """Create all tables if they don't already exist. Idempotent — safe to
    call on every app startup. The real Postgres deployment applies
    sql/schema.sql directly (see docker-compose.yml); this exists mainly so
    SQLite-backed tests and ad hoc local scripts don't need a separate
    migration step.
    """
    Base.metadata.create_all(engine or create_db_engine())


@contextmanager
def session_scope(session_factory: sessionmaker | None = None) -> Iterator[Session]:
    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
