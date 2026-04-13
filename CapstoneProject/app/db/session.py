"""SQLAlchemy session factory and database initialisation."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models.database import Base

engine = create_engine(
    settings.database_url,
    connect_args={
        "check_same_thread": False,  # required for SQLite
        "timeout": 30,               # wait up to 30 s when DB is locked
    },
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a database session and ensures cleanup."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
