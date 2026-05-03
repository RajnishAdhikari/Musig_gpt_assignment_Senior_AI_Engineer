from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


VECTOR_DIMENSIONS = 384


SEARCH_VECTOR_SQL = """
setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
setweight(
    to_tsvector(
        'english',
        coalesce(musical_key, '') || ' ' ||
        coalesce(bpm::text, '') || ' bpm ' ||
        coalesce(duration_type, '') || ' ' ||
        coalesce(primary_genre, '') || ' ' ||
        coalesce(primary_mood, '') || ' ' ||
        coalesce(array_to_string(vocal_genders, ' '), '')
    ),
    'A'
) ||
setweight(
    to_tsvector(
        'english',
        coalesce(prompt, '') || ' ' || coalesce(acoustic_prompt_descriptive, '')
    ),
    'B'
) ||
setweight(to_tsvector('english', coalesce(sounds, '')), 'C') ||
setweight(to_tsvector('english', coalesce(array_to_string(all_tags, ' '), '')), 'D')
"""


class Base(DeclarativeBase):
    pass


class Generation(Base):
    __tablename__ = "generations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(Text)
    prompt: Mapped[Optional[str]] = mapped_column(Text)
    lyrics: Mapped[Optional[str]] = mapped_column(Text)
    num_outputs: Mapped[Optional[int]] = mapped_column(Integer)
    raw_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    search_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    songs: Mapped[list["Song"]] = relationship(
        back_populates="generation",
        cascade="all, delete-orphan",
    )


class Song(Base):
    __tablename__ = "songs"
    __table_args__ = (
        CheckConstraint("variant_index in (1, 2)", name="ck_songs_variant_index"),
        CheckConstraint("clicks >= 0", name="ck_songs_clicks_nonnegative"),
        CheckConstraint("impressions >= 0", name="ck_songs_impressions_nonnegative"),
        UniqueConstraint("lineage_id", "variant_index", name="uq_songs_lineage_variant"),
        Index("ix_songs_lineage_id", "lineage_id"),
        Index("ix_songs_created_at", text("created_at DESC")),
        Index("ix_songs_primary_genre", "primary_genre"),
        Index("ix_songs_bpm", "bpm"),
        Index("ix_songs_musical_key", "musical_key"),
        Index("ix_songs_search_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_songs_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    lineage_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("generations.id", ondelete="CASCADE"),
        nullable=False,
    )
    variant_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    conversion_path: Mapped[str] = mapped_column(Text, nullable=False)

    title: Mapped[Optional[str]] = mapped_column(Text)
    prompt: Mapped[Optional[str]] = mapped_column(Text)
    sounds: Mapped[Optional[str]] = mapped_column(Text)
    lyrics: Mapped[Optional[str]] = mapped_column(Text)
    acoustic_prompt_descriptive: Mapped[Optional[str]] = mapped_column(Text)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False)

    bpm: Mapped[Optional[int]] = mapped_column(Integer)
    musical_key: Mapped[Optional[str]] = mapped_column(Text)
    duration_type: Mapped[Optional[str]] = mapped_column(Text)
    primary_genre: Mapped[Optional[str]] = mapped_column(Text)
    primary_mood: Mapped[Optional[str]] = mapped_column(Text)
    vocal_genders: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )
    all_tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("'{}'::text[]"),
    )
    search_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False)

    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(VECTOR_DIMENSIONS))
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        nullable=False,
        server_default=text("''::tsvector"),
    )

    clicks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    impressions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("0"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    generation: Mapped[Generation] = relationship(back_populates="songs")
    feedback_events: Mapped[list["FeedbackEvent"]] = relationship(back_populates="song")


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"
    __table_args__ = (
        CheckConstraint(
            "event_type in ('click', 'impression')",
            name="ck_feedback_events_event_type",
        ),
        Index("ix_feedback_events_output_id", "output_id"),
        Index(
            "ix_feedback_events_unprocessed",
            "received_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    output_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("songs.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    flush_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))

    song: Mapped[Song] = relationship(back_populates="feedback_events")
