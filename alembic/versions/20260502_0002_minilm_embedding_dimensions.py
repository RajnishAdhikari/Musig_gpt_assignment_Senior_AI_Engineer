"""use MiniLM embedding dimensions

Revision ID: 20260502_0002
Revises: 20260429_0001
Create Date: 2026-05-02
"""

from __future__ import annotations

from alembic import op


revision = "20260502_0002"
down_revision = "20260429_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_songs_embedding_hnsw", table_name="songs")
    op.execute("UPDATE songs SET embedding = NULL")
    op.execute("ALTER TABLE songs ALTER COLUMN embedding TYPE vector(384) USING NULL::vector(384)")
    op.execute(
        """
        CREATE INDEX ix_songs_embedding_hnsw
        ON songs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.drop_index("ix_songs_embedding_hnsw", table_name="songs")
    op.execute("UPDATE songs SET embedding = NULL")
    op.execute("ALTER TABLE songs ALTER COLUMN embedding TYPE vector(384) USING NULL::vector(384)")
    op.execute(
        """
        CREATE INDEX ix_songs_embedding_hnsw
        ON songs
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
