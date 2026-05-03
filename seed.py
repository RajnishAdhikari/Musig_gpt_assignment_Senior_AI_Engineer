from __future__ import annotations

import argparse
import asyncio
import json
import re
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from db import async_session_factory, engine
from models import FeedbackEvent, Generation, Song
from ranking import calculate_final_score, demo_score_table, diversify_results
from search import embed_text, hybrid_search, vector_only_search
from settings import settings


DATASET_URL = settings.dataset_url
DATASET_PATH = settings.dataset_path
FRESH_POP_LINEAGE_ID = uuid.UUID("29d8590d-9c4d-4c60-ad7b-d8138ad333f5")
OLD_POP_LINEAGE_ID = uuid.UUID("14c8d840-ae07-4b92-be72-f24adf3e11bc")
LOW_CONFIDENCE_LINEAGE_ID = uuid.UUID("0a43dfaf-f308-4df7-803d-02384f471c5e")
COLD_START_LINEAGE_ID = uuid.UUID("e0a6c2ce-8c5b-4bd3-a151-2128c0d0d537")
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
DYNAMODB_KEYS = {"S", "N", "BOOL", "NULL", "L", "M", "SS", "NS"}


def unwrap_dynamodb(value: Any) -> Any:
    if isinstance(value, list):
        return [unwrap_dynamodb(item) for item in value]
    if not isinstance(value, dict):
        return value

    keys = set(value)
    if len(keys & DYNAMODB_KEYS) == 1 and len(keys) == 1:
        if "S" in value:
            return value["S"]
        if "N" in value:
            number = value["N"]
            try:
                return int(number) if str(number).lstrip("-").isdigit() else float(number)
            except (TypeError, ValueError):
                return number
        if "BOOL" in value:
            return bool(value["BOOL"])
        if "NULL" in value:
            return None
        if "L" in value:
            return [unwrap_dynamodb(item) for item in (value["L"] or [])]
        if "M" in value:
            return {key: unwrap_dynamodb(item) for key, item in (value["M"] or {}).items()}
        if "SS" in value:
            return list(value["SS"] or [])
        if "NS" in value:
            return [unwrap_dynamodb({"N": item}) for item in (value["NS"] or [])]

    return {key: unwrap_dynamodb(item) for key, item in value.items()}


def ensure_dataset() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        try:
            json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            return
        except json.JSONDecodeError:
            DATASET_PATH.unlink()

    print(f"Downloading dataset to {DATASET_PATH}...")
    with urllib.request.urlopen(DATASET_URL, timeout=settings.dataset_download_timeout_seconds) as response:
        DATASET_PATH.write_bytes(response.read())


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def output_uuid(conversion_path: str, lineage_id: uuid.UUID, variant_index: int) -> uuid.UUID:
    matches = UUID_RE.findall(conversion_path or "")
    if matches:
        return uuid.UUID(matches[-1])
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{lineage_id}:{variant_index}:{conversion_path}")


def choose_embedding_text(
    acoustic_prompt_descriptive: str | None,
    sounds: str | None,
    prompt: str | None,
    title: str | None,
    all_tags: list[str],
) -> str:
    if acoustic_prompt_descriptive:
        return acoustic_prompt_descriptive
    fallback_parts = [sounds or "", prompt or "", title or "", " ".join(all_tags)]
    return " ".join(part for part in fallback_parts if part).strip() or "untitled music generation"


def synthetic_metrics(
    lineage_id: uuid.UUID,
    record_index: int,
    variant_index: int,
    now: datetime,
) -> tuple[datetime, int, int]:
    if lineage_id == FRESH_POP_LINEAGE_ID:
        age_days, clicks, impressions = 3, 40, 60
    elif lineage_id == OLD_POP_LINEAGE_ID:
        age_days, clicks, impressions = 730, 1000, 5000
    elif lineage_id == LOW_CONFIDENCE_LINEAGE_ID:
        age_days, clicks, impressions = 1, 1, 1
    elif lineage_id == COLD_START_LINEAGE_ID:
        age_days, clicks, impressions = 183, 0, 0
    else:
        age_days = 14 + ((record_index * 23) % 620)
        impressions = (record_index * 37) % 450
        ctr_basis = 0.03 + ((record_index % 7) * 0.025)
        clicks = min(impressions, int(impressions * ctr_basis))

    if variant_index == 2 and impressions > 0:
        clicks = max(clicks // 2, 0)
    return now - timedelta(days=age_days), clicks, impressions


async def clear_existing(session: AsyncSession) -> None:
    await session.execute(delete(FeedbackEvent))
    await session.execute(delete(Song))
    await session.execute(delete(Generation))


async def seed_database(session: AsyncSession, *, max_records: int | None = None) -> int:
    ensure_dataset()
    raw_rows = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("expected the dataset JSON root to be a list")

    await clear_existing(session)
    now = datetime.now(timezone.utc)
    rows = raw_rows[:max_records] if max_records else raw_rows

    for record_index, raw_row in enumerate(rows):
        lineage_id = uuid.UUID(str(raw_row["id"]))
        metadata = unwrap_dynamodb(raw_row.get("search_metadata") or {})
        technical = metadata.get("technical") or {}
        core = metadata.get("core_attributes") or {}
        genre = core.get("genre") or {}
        mood = core.get("mood") or {}
        vocals = core.get("vocals") or {}
        all_tags = as_str_list(metadata.get("all_tags"))
        acoustic_prompt_descriptive = metadata.get("acoustic_prompt_descriptive")

        generation = Generation(
            id=lineage_id,
            title=raw_row.get("title"),
            prompt=raw_row.get("prompt"),
            lyrics=raw_row.get("lyrics"),
            num_outputs=to_int(raw_row.get("num_outputs")),
            raw_payload=raw_row,
            search_metadata=metadata,
        )
        session.add(generation)

        for variant_index in (1, 2):
            conversion_path = raw_row.get(f"conversion_path_{variant_index}")
            if not conversion_path:
                continue
            sounds = raw_row.get(f"sounds_{variant_index}")
            lyrics = raw_row.get(f"lyrics_{variant_index}") or raw_row.get("lyrics")
            embedding_text = choose_embedding_text(
                acoustic_prompt_descriptive,
                sounds,
                raw_row.get("prompt"),
                raw_row.get("title"),
                all_tags,
            )
            created_at, clicks, impressions = synthetic_metrics(
                lineage_id,
                record_index,
                variant_index,
                now,
            )

            session.add(
                Song(
                    id=output_uuid(conversion_path, lineage_id, variant_index),
                    lineage_id=lineage_id,
                    variant_index=variant_index,
                    conversion_path=conversion_path,
                    title=raw_row.get("title"),
                    prompt=metadata.get("prompt") or raw_row.get("prompt"),
                    sounds=sounds,
                    lyrics=lyrics,
                    acoustic_prompt_descriptive=acoustic_prompt_descriptive,
                    embedding_text=embedding_text,
                    bpm=to_int(technical.get("bpm")),
                    musical_key=technical.get("key"),
                    duration_type=technical.get("duration_type"),
                    primary_genre=genre.get("primary_genre"),
                    primary_mood=mood.get("primary_mood"),
                    vocal_genders=as_str_list(vocals.get("vocal_gender")),
                    all_tags=all_tags,
                    search_metadata=metadata,
                    embedding=embed_text(embedding_text),
                    clicks=clicks,
                    impressions=impressions,
                    created_at=created_at,
                )
            )

    return len(rows)


def print_score_demo() -> None:
    print("\nPart 3 score sanity check")
    print("Song Age Clicks Impressions Hybrid Engagement Recency Final")
    for row in demo_score_table():
        print(
            f"{row['song']:>4} {row['age_days']:>3} {row['clicks']:>6} "
            f"{row['impressions']:>11} {row['hybrid_score']:>6.2f} "
            f"{row['engagement_score']:>10.4f} {row['recency_score']:>7.4f} "
            f"{row['final_score']:>7.4f}"
        )


def print_ranked(title: str, rows: list[dict[str, Any]], *, limit: int = 8) -> None:
    print(f"\n{title}")
    print("pos title                  var age clicks/impr hybrid  final   v_rank f_rank lineage")
    now = datetime.now(timezone.utc)
    for index, row in enumerate(rows[:limit], start=1):
        age_days = int((now - row["created_at"]).total_seconds() / 86_400)
        print(
            f"{index:>3} {row['title'][:22]:<22} {row['variant_index']:>3} "
            f"{age_days:>3} {row['clicks']:>5}/{row['impressions']:<5} "
            f"{row.get('hybrid_score', 0):>7.5f} {row.get('final_score', 0):>7.5f} "
            f"{str(row.get('vector_rank') or '-'):>6} {str(row.get('fts_rank') or '-'):>6} "
            f"{str(row['lineage_id'])[:8]}"
        )


async def verify(session: AsyncSession) -> None:
    print_score_demo()

    new_pop = await hybrid_search(session, "new pop", limit=60, retrieval_limit=100)
    for row in new_pop:
        row["final_score"] = calculate_final_score(row, row["hybrid_score"])
    new_pop = sorted(new_pop, key=lambda item: item["final_score"], reverse=True)
    new_pop_diverse = diversify_results(new_pop)
    print_ranked('Verification query: "new pop" after re-ranking + diversity', new_pop_diverse)

    positions: dict[uuid.UUID, int] = {}
    for index, row in enumerate(new_pop_diverse, start=1):
        positions.setdefault(row["lineage_id"], index)
    print(
        "\nExpected ordering check: "
        f"fresh 3-day pop lineage position={positions.get(FRESH_POP_LINEAGE_ID)}, "
        f"old 2-year pop lineage position={positions.get(OLD_POP_LINEAGE_ID)}"
    )
    for label, lineage_id in (
        ("fresh", FRESH_POP_LINEAGE_ID),
        ("old", OLD_POP_LINEAGE_ID),
    ):
        row = next(item for item in new_pop_diverse if item["lineage_id"] == lineage_id)
        age_days = int((datetime.now(timezone.utc) - row["created_at"]).total_seconds() / 86_400)
        print(
            f"{label.title()} case detail: pos={positions[lineage_id]} "
            f"title={row['title']} var={row['variant_index']} age={age_days}d "
            f"clicks/impressions={row['clicks']}/{row['impressions']} "
            f"hybrid={row['hybrid_score']:.5f} final={row['final_score']:.5f}"
        )

    exact_query = "C major female vocal"
    vector_rows = await vector_only_search(session, exact_query, limit=5)
    hybrid_rows = await hybrid_search(session, exact_query, limit=12, retrieval_limit=100)
    for row in hybrid_rows:
        row["final_score"] = calculate_final_score(row, row["hybrid_score"])

    print(f"\nVector-only top 5 for \"{exact_query}\"")
    for index, row in enumerate(vector_rows, start=1):
        print(
            f"{index:>3} {row['title']:<22} var={row['variant_index']} "
            f"vector_score={row['vector_score']:.4f} key={row['musical_key']} "
            f"vocals={','.join(row['vocal_genders'])}"
        )

    print_ranked(f'Hybrid top results for "{exact_query}"', hybrid_rows, limit=8)
    vector_ids = {row["id"] for row in vector_rows}
    lexical_only = next(
        (row for row in hybrid_rows if row["id"] not in vector_ids and row.get("fts_rank") is not None),
        None,
    )
    if lexical_only:
        print(
            "\nLexical rescue: "
            f"{lexical_only['title']} var={lexical_only['variant_index']} "
            f"has fts_rank={lexical_only['fts_rank']} and was absent from vector-only top 5."
        )


async def async_main(args: argparse.Namespace) -> None:
    async with async_session_factory() as session:
        async with session.begin():
            count = await seed_database(session, max_records=args.max_records)
        print(f"Seeded {count} generation records ({count * 2} song outputs).")

    if not args.no_verify:
        async with async_session_factory() as session:
            await verify(session)
    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
