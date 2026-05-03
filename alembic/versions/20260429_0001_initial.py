"""initial search schema

Revision ID: 20260429_0001
Revises:
Create Date: 2026-04-29
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision = "20260429_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "generations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("lyrics", sa.Text(), nullable=True),
        sa.Column("num_outputs", sa.Integer(), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "songs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("lineage_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("variant_index", sa.SmallInteger(), nullable=False),
        sa.Column("conversion_path", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("sounds", sa.Text(), nullable=True),
        sa.Column("lyrics", sa.Text(), nullable=True),
        sa.Column("acoustic_prompt_descriptive", sa.Text(), nullable=True),
        sa.Column("embedding_text", sa.Text(), nullable=False),
        sa.Column("bpm", sa.Integer(), nullable=True),
        sa.Column("musical_key", sa.Text(), nullable=True),
        sa.Column("duration_type", sa.Text(), nullable=True),
        sa.Column("primary_genre", sa.Text(), nullable=True),
        sa.Column("primary_mood", sa.Text(), nullable=True),
        sa.Column("vocal_genders", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
        sa.Column("all_tags", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'::text[]"), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("search_vector", postgresql.TSVECTOR(), server_default=sa.text("''::tsvector"), nullable=False),
        sa.Column("clicks", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("impressions", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("clicks >= 0", name="ck_songs_clicks_nonnegative"),
        sa.CheckConstraint("impressions >= 0", name="ck_songs_impressions_nonnegative"),
        sa.CheckConstraint("variant_index in (1, 2)", name="ck_songs_variant_index"),
        sa.ForeignKeyConstraint(["lineage_id"], ["generations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lineage_id", "variant_index", name="uq_songs_lineage_variant"),
    )

    op.execute(
        """
        CREATE FUNCTION songs_search_vector_refresh() RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            NEW.search_vector :=
                setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
                setweight(
                    to_tsvector(
                        'english',
                        coalesce(NEW.musical_key, '') || ' ' ||
                        coalesce(NEW.bpm::text, '') || ' bpm ' ||
                        coalesce(NEW.duration_type, '') || ' ' ||
                        coalesce(NEW.primary_genre, '') || ' ' ||
                        coalesce(NEW.primary_mood, '') || ' ' ||
                        coalesce(array_to_string(NEW.vocal_genders, ' '), '')
                    ),
                    'A'
                ) ||
                setweight(
                    to_tsvector(
                        'english',
                        coalesce(NEW.prompt, '') || ' ' ||
                        coalesce(NEW.acoustic_prompt_descriptive, '')
                    ),
                    'B'
                ) ||
                setweight(to_tsvector('english', coalesce(NEW.sounds, '')), 'C') ||
                setweight(
                    to_tsvector('english', coalesce(array_to_string(NEW.all_tags, ' '), '')),
                    'D'
                );
            RETURN NEW;
        END;
        $$;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_songs_search_vector_refresh
        BEFORE INSERT OR UPDATE OF
            title,
            prompt,
            acoustic_prompt_descriptive,
            sounds,
            musical_key,
            bpm,
            duration_type,
            primary_genre,
            primary_mood,
            vocal_genders,
            all_tags
        ON songs
        FOR EACH ROW
        EXECUTE FUNCTION songs_search_vector_refresh()
        """
    )

    op.create_table(
        "feedback_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("output_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("flush_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.CheckConstraint("event_type in ('click', 'impression')", name="ck_feedback_events_event_type"),
        sa.ForeignKeyConstraint(["output_id"], ["songs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_songs_lineage_id", "songs", ["lineage_id"])
    op.create_index("ix_songs_created_at", "songs", [sa.text("created_at DESC")])
    op.create_index("ix_songs_primary_genre", "songs", ["primary_genre"])
    op.create_index("ix_songs_bpm", "songs", ["bpm"])
    op.create_index("ix_songs_musical_key", "songs", ["musical_key"])
    op.create_index("ix_songs_search_vector", "songs", ["search_vector"], postgresql_using="gin")
    op.create_index(
        "ix_songs_metadata_gin",
        "songs",
        ["metadata"],
        postgresql_using="gin",
        postgresql_ops={"metadata": "jsonb_path_ops"},
    )
    op.execute(
        """
        CREATE INDEX ix_songs_embedding_hnsw
        ON songs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    op.create_index("ix_feedback_events_output_id", "feedback_events", ["output_id"])
    op.create_index(
        "ix_feedback_events_unprocessed",
        "feedback_events",
        ["received_at"],
        postgresql_where=sa.text("processed_at IS NULL"),
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_songs_search_vector_refresh ON songs")
    op.execute("DROP FUNCTION IF EXISTS songs_search_vector_refresh()")
    op.drop_index("ix_feedback_events_unprocessed", table_name="feedback_events")
    op.drop_index("ix_feedback_events_output_id", table_name="feedback_events")
    op.drop_index("ix_songs_embedding_hnsw", table_name="songs")
    op.drop_index("ix_songs_metadata_gin", table_name="songs")
    op.drop_index("ix_songs_search_vector", table_name="songs")
    op.drop_index("ix_songs_musical_key", table_name="songs")
    op.drop_index("ix_songs_bpm", table_name="songs")
    op.drop_index("ix_songs_primary_genre", table_name="songs")
    op.drop_index("ix_songs_created_at", table_name="songs")
    op.drop_index("ix_songs_lineage_id", table_name="songs")
    op.drop_table("feedback_events")
    op.drop_table("songs")
    op.drop_table("generations")
