"""
Database session management and table initialization for API Gateway (`services/api-gateway`).
"""

from shared.config import get_settings
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

_engine: Engine | None = None
_sync_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Returns singleton SQLAlchemy Engine for Postgres connection."""
    global _engine
    if _engine is None:
        settings = get_settings()
        dsn = settings.POSTGRES_DSN
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = dsn.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        elif dsn.startswith("postgresql://") and "+psycopg2" not in dsn:
            dsn = dsn.replace("postgresql://", "postgresql+psycopg2://")
        _engine = create_engine(dsn, pool_pre_ping=True, echo=False)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Returns singleton sessionmaker factory."""
    global _sync_session_factory
    if _sync_session_factory is None:
        engine = get_engine()
        _sync_session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return _sync_session_factory


def get_sync_session() -> Session:
    """Returns a new synchronous SQLAlchemy session."""
    factory = get_session_factory()
    return factory()


def init_db() -> None:
    """Synchronously creates required database tables (`document_permissions`, etc)."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
