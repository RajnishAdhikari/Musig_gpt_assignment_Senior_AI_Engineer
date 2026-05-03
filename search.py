from __future__ import annotations

import re
import os
from functools import lru_cache
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from models import VECTOR_DIMENSIONS
from settings import settings


# MiniLM is loaded through sentence-transformers, which uses PyTorch here.
# Disable optional backends before transformers is imported so a broken local
# TensorFlow/Flax installation cannot prevent the search service from starting.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Exact technical terms remain in FTS, while the semantic embedding focuses on
# descriptive/vibe text. This keeps keyword precision and vector semantics from
# fighting each other for queries such as "C major female vocal".
SEMANTIC_EXACT_TERMS = {
    "a",
    "an",
    "and",
    "at",
    "bpm",
    "c",
    "d",
    "e",
    "f",
    "g",
    "major",
    "minor",
    "male",
    "female",
    "mixed",
    "vocal",
    "vocals",
    "voice",
    "singing",
}


@lru_cache(maxsize=1)
def embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Failed to load sentence-transformers for embeddings. "
            "Install dependencies with `python -m pip install -r requirements.txt`; "
            "if it is already installed, check the local transformers/PyTorch backend."
        ) from exc

    return SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)


HYBRID_SEARCH_SQL = """
WITH parsed_query AS (
    SELECT
        CAST(:query_embedding AS vector) AS query_embedding,
        websearch_to_tsquery('english', :query_text) AS fts_query
),
vector_ranked AS (
    SELECT
        s.id,
        ROW_NUMBER() OVER (ORDER BY s.embedding <=> q.query_embedding, s.id) AS vector_rank
    FROM songs s
    CROSS JOIN parsed_query q
    WHERE s.embedding IS NOT NULL
    ORDER BY s.embedding <=> q.query_embedding, s.id
    LIMIT :retrieval_limit
),
fts_ranked AS (
    SELECT
        s.id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(s.search_vector, q.fts_query, 32) DESC, s.id
        ) AS fts_rank
    FROM songs s
    CROSS JOIN parsed_query q
    WHERE numnode(q.fts_query) > 0
      AND s.search_vector @@ q.fts_query
    ORDER BY ts_rank_cd(s.search_vector, q.fts_query, 32) DESC, s.id
    LIMIT :retrieval_limit
),
fused AS (
    SELECT
        COALESCE(v.id, f.id) AS id,
        COALESCE(
            CAST(:vector_weight AS double precision) /
            (CAST(:rrf_k AS double precision) + v.vector_rank),
            0.0
        ) +
        COALESCE(
            CAST(:fts_weight AS double precision) /
            (CAST(:rrf_k AS double precision) + f.fts_rank),
            0.0
        ) AS hybrid_score
    FROM vector_ranked v
    FULL OUTER JOIN fts_ranked f ON v.id = f.id
)
SELECT
    s.id,
    s.lineage_id,
    s.variant_index,
    s.conversion_path,
    s.title,
    s.prompt,
    s.sounds,
    s.acoustic_prompt_descriptive,
    s.bpm,
    s.musical_key,
    s.duration_type,
    s.primary_genre,
    s.primary_mood,
    s.vocal_genders,
    s.all_tags,
    s.clicks,
    s.impressions,
    s.created_at,
    fused.hybrid_score,
    v.vector_rank,
    f.fts_rank
FROM fused
JOIN songs s ON s.id = fused.id
LEFT JOIN vector_ranked v ON v.id = s.id
LEFT JOIN fts_ranked f ON f.id = s.id
ORDER BY fused.hybrid_score DESC, s.id
LIMIT :limit
"""


VECTOR_ONLY_SQL = """
SELECT
    s.id,
    s.lineage_id,
    s.variant_index,
    s.title,
    s.primary_genre,
    s.primary_mood,
    s.bpm,
    s.musical_key,
    s.vocal_genders,
    s.clicks,
    s.impressions,
    s.created_at,
    1.0 - (s.embedding <=> CAST(:query_embedding AS vector)) AS vector_score
FROM songs s
WHERE s.embedding IS NOT NULL
ORDER BY s.embedding <=> CAST(:query_embedding AS vector), s.id
LIMIT :limit
"""


def tokenize(text_value: str) -> list[str]:
    return TOKEN_RE.findall(text_value.lower())


@lru_cache(maxsize=4096)
def embed_text(text_value: str, dimensions: int = VECTOR_DIMENSIONS) -> list[float]:
    tokens = tokenize(text_value)
    semantic_tokens = [token for token in tokens if token not in SEMANTIC_EXACT_TERMS]
    if not semantic_tokens:
        semantic_tokens = tokens or ["empty"]

    embedding_input = " ".join(semantic_tokens)
    vector = embedding_model().encode(
        embedding_input,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    values = [float(value) for value in vector]
    if len(values) != dimensions:
        raise ValueError(
            f"{settings.embedding_model_name} returned {len(values)} dimensions; "
            f"expected {dimensions}. Update VECTOR_DIMENSIONS and the pgvector migration."
        )
    return values


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    *,
    limit: int = settings.default_search_limit,
    retrieval_limit: int = settings.default_retrieval_limit,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    fts_weight: float = 1.15,
) -> list[dict[str, Any]]:
    query_embedding = vector_literal(embed_text(query_text))
    result = await session.execute(
        text(HYBRID_SEARCH_SQL),
        {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "limit": limit,
            "retrieval_limit": retrieval_limit,
            "rrf_k": rrf_k,
            "vector_weight": vector_weight,
            "fts_weight": fts_weight,
        },
    )
    return [dict(row) for row in result.mappings()]


async def vector_only_search(
    session: AsyncSession,
    query_text: str,
    *,
    limit: int = settings.default_search_limit,
) -> list[dict[str, Any]]:
    query_embedding = vector_literal(embed_text(query_text))
    result = await session.execute(
        text(VECTOR_ONLY_SQL),
        {"query_embedding": query_embedding, "limit": limit},
    )
    return [dict(row) for row in result.mappings()]
