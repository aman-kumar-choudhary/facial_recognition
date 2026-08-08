from datetime import datetime

from sqlalchemy import String, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Student(Base):
    """Primary metadata record. The embedding itself is NOT stored here --
    it lives in the vector store, referenced by `student_id`."""

    __tablename__ = "students"

    # The roll number is the stable public identifier used everywhere.
    # Do not generate a second opaque UUID for a student.
    student_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    roll_number: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(32), default="student")
    embedding_quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
