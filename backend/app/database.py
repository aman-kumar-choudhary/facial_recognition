"""
Async SQLAlchemy engine + session factory for the primary metadata store.
This holds student identity records; embeddings live in the vector store,
not here (see app/vector_store.py) -- keeps the metadata DB lightweight
and lets the embedding index scale/be swapped independently.
"""
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, text
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger(__name__)

os.makedirs("./data", exist_ok=True)

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def init_db() -> dict[str, str]:
    from app.models_db import Student
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Lightweight compatibility migration for deployments created before
        # face updates had an explicit timestamp.
        columns = (await conn.execute(text("PRAGMA table_info(students)"))).all()
        if "updated_at" not in {row[1] for row in columns}:
            await conn.execute(text("ALTER TABLE students ADD COLUMN updated_at DATETIME"))
            await conn.execute(text("UPDATE students SET updated_at = created_at WHERE updated_at IS NULL"))
    # `create_all` does not migrate existing rows. Safely replace legacy UUID
    # primary keys with their already-unique roll numbers during startup.
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(Student.student_id, Student.roll_number))).all()
        migrations = {student_id: roll_number for student_id, roll_number in rows if student_id != roll_number}
        for old_id, roll_number in migrations.items():
            await session.execute(update(Student).where(Student.student_id == old_id).values(student_id=roll_number))
        if migrations:
            await session.commit()
            logger.info("student_ids_migrated_to_roll_numbers", extra={"event": "student_ids_migrated_to_roll_numbers", "count": len(migrations)})
    logger.info("database_initialized", extra={"event": "database_initialized"})
    return migrations


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
