from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models import FeedbackEvent
from settings import settings


FeedbackType = Literal["click", "impression"]
DEFAULT_FLUSH_INTERVAL_SECONDS = settings.feedback_flush_interval_seconds
DEFAULT_FLUSH_BATCH_SIZE = settings.feedback_flush_batch_size


@dataclass(frozen=True)
class FlushResult:
    flush_id: uuid.UUID
    events_processed: int
    songs_touched: int


async def record_feedback(
    session: AsyncSession,
    *,
    output_id: uuid.UUID,
    event_type: FeedbackType,
    event_id: uuid.UUID | None = None,
) -> uuid.UUID:
    if event_type not in ("click", "impression"):
        raise ValueError(f"unsupported feedback type: {event_type}")

    feedback_id = event_id or uuid.uuid4()
    statement = (
        insert(FeedbackEvent)
        .values(
            id=feedback_id,
            output_id=output_id,
            event_type=event_type,
        )
        .on_conflict_do_nothing(index_elements=[FeedbackEvent.id])
    )
    await session.execute(statement)
    return feedback_id


async def flush_feedback_events(
    session: AsyncSession,
    *,
    batch_size: int = DEFAULT_FLUSH_BATCH_SIZE,
) -> FlushResult:
    flush_id = uuid.uuid4()
    result = await session.execute(
        text(
            """
            WITH batch AS (
                SELECT id, output_id, event_type
                FROM feedback_events
                WHERE processed_at IS NULL
                ORDER BY received_at
                LIMIT :batch_size
                FOR UPDATE SKIP LOCKED
            ),
            aggregated AS (
                SELECT
                    output_id,
                    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END)::int AS clicks,
                    SUM(CASE WHEN event_type = 'impression' THEN 1 ELSE 0 END)::int AS impressions
                FROM batch
                GROUP BY output_id
            ),
            updated AS (
                UPDATE songs s
                SET
                    clicks = s.clicks + a.clicks,
                    impressions = s.impressions + a.impressions,
                    updated_at = now()
                FROM aggregated a
                WHERE s.id = a.output_id
                RETURNING s.id
            ),
            marked AS (
                UPDATE feedback_events fe
                SET processed_at = now(), flush_id = :flush_id
                WHERE fe.id IN (SELECT id FROM batch)
                RETURNING fe.id
            )
            SELECT
                (SELECT count(*) FROM marked)::int AS events_processed,
                (SELECT count(*) FROM updated)::int AS songs_touched
            """
        ),
        {
            "batch_size": batch_size,
            "flush_id": flush_id,
        },
    )
    row = result.one()
    return FlushResult(
        flush_id=flush_id,
        events_processed=row.events_processed,
        songs_touched=row.songs_touched,
    )


async def run_flush_loop(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
    batch_size: int = DEFAULT_FLUSH_BATCH_SIZE,
) -> None:
    while True:
        async with session_factory() as session:
            async with session.begin():
                await flush_feedback_events(session, batch_size=batch_size)
        await asyncio.sleep(interval_seconds)
