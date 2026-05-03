from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, TypeVar


PRIOR_CTR = 0.08
CONFIDENCE_IMPRESSIONS = 50.0
RECENCY_HALF_LIFE_DAYS = 180.0
RECENCY_FLOOR = 0.70
RECENCY_RANGE = 0.30
ENGAGEMENT_FLOOR = 0.85
ENGAGEMENT_RANGE = 0.50
DUPLICATE_LINEAGE_PENALTY = 0.72

T = TypeVar("T")


def _get(song: Any, key: str, default: Any = None) -> Any:
    if isinstance(song, Mapping):
        return song.get(key, default)
    return getattr(song, key, default)


def _age_days(song: Any, now: datetime) -> float | None:
    explicit_age = _get(song, "age_days")
    if explicit_age is not None:
        return max(float(explicit_age), 0.0)

    created_at = _get(song, "created_at")
    if not isinstance(created_at, datetime):
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return max((now - created_at).total_seconds() / 86_400.0, 0.0)


def wilson_lower_bound(clicks: int, impressions: int, z: float = 1.28) -> float:
    if impressions <= 0:
        return PRIOR_CTR
    p_hat = clicks / impressions
    denominator = 1.0 + (z * z / impressions)
    center = p_hat + (z * z / (2.0 * impressions))
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * impressions)) / impressions)
    return max(0.0, (center - margin) / denominator)


def engagement_score(clicks: int, impressions: int) -> float:
    safe_impressions = max(int(impressions or 0), 0)
    safe_clicks = max(int(clicks or 0), 0)
    if safe_clicks > safe_impressions:
        safe_impressions = safe_clicks

    confidence = 1.0 - math.exp(-safe_impressions / CONFIDENCE_IMPRESSIONS)
    conservative_ctr = wilson_lower_bound(safe_clicks, safe_impressions)
    return (confidence * conservative_ctr) + ((1.0 - confidence) * PRIOR_CTR)


def recency_score(age_days: float | None) -> float:
    if age_days is None:
        return 0.5
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def calculate_final_score(
    song: Any,
    hybrid_score: float,
    *,
    now: datetime | None = None,
) -> float:
    now = now or datetime.now(timezone.utc)
    clicks = max(int(_get(song, "clicks", 0) or 0), 0)
    impressions = max(int(_get(song, "impressions", 0) or 0), 0)
    age = _age_days(song, now)

    recency_multiplier = RECENCY_FLOOR + (RECENCY_RANGE * recency_score(age))
    engagement_multiplier = ENGAGEMENT_FLOOR + (
        ENGAGEMENT_RANGE * engagement_score(clicks, impressions)
    )
    return float(hybrid_score) * recency_multiplier * engagement_multiplier


def _lineage_id(item: Any) -> str:
    value = _get(item, "lineage_id", _get(item, "generation_id", _get(item, "id")))
    return str(value)


def _score(item: Any) -> float:
    return float(_get(item, "final_score", _get(item, "score", 0.0)) or 0.0)


def _set_adjusted_score(item: T, value: float) -> T:
    if isinstance(item, dict):
        copied = dict(item)
        copied["diversified_score"] = value
        return copied  # type: ignore[return-value]
    return item


def diversify_results(
    ranked_list: Sequence[T],
    *,
    top_k: int = 5,
    min_distinct_lineages: int = 4,
    duplicate_penalty: float = DUPLICATE_LINEAGE_PENALTY,
) -> list[T]:
    remaining = list(ranked_list)
    if len(remaining) <= 1:
        return remaining

    available_lineages = {_lineage_id(item) for item in remaining}
    target_distinct = min(min_distinct_lineages, top_k, len(available_lineages))
    chosen: list[T] = []
    seen_counts: dict[str, int] = {}

    while remaining and len(chosen) < min(top_k, len(ranked_list)):
        remaining_slots = top_k - len(chosen)
        distinct_needed = max(target_distinct - len(seen_counts), 0)
        must_pick_new_lineage = distinct_needed >= remaining_slots

        best_index = 0
        best_score = -1.0
        for index, item in enumerate(remaining):
            lineage = _lineage_id(item)
            if must_pick_new_lineage and lineage in seen_counts:
                continue
            penalty_power = seen_counts.get(lineage, 0)
            adjusted_score = _score(item) * (duplicate_penalty ** penalty_power)
            if adjusted_score > best_score:
                best_index = index
                best_score = adjusted_score

        item = remaining.pop(best_index)
        chosen.append(_set_adjusted_score(item, best_score))
        lineage = _lineage_id(item)
        seen_counts[lineage] = seen_counts.get(lineage, 0) + 1

    while remaining:
        best_index = 0
        best_score = -1.0
        for index, item in enumerate(remaining):
            lineage = _lineage_id(item)
            adjusted_score = _score(item) * (duplicate_penalty ** seen_counts.get(lineage, 0))
            if adjusted_score > best_score:
                best_index = index
                best_score = adjusted_score
        item = remaining.pop(best_index)
        chosen.append(_set_adjusted_score(item, best_score))
        lineage = _lineage_id(item)
        seen_counts[lineage] = seen_counts.get(lineage, 0) + 1

    return chosen


def demo_score_table() -> list[dict[str, float | int | str]]:
    cases = [
        {"song": "A", "age_days": 3, "clicks": 40, "impressions": 60, "hybrid_score": 0.72},
        {"song": "B", "age_days": 730, "clicks": 1000, "impressions": 5000, "hybrid_score": 0.80},
        {"song": "C", "age_days": 1, "clicks": 1, "impressions": 1, "hybrid_score": 0.75},
        {"song": "D", "age_days": 183, "clicks": 0, "impressions": 0, "hybrid_score": 0.68},
    ]
    for case in cases:
        case["final_score"] = round(calculate_final_score(case, float(case["hybrid_score"])), 6)
        case["engagement_score"] = round(
            engagement_score(int(case["clicks"]), int(case["impressions"])),
            6,
        )
        case["recency_score"] = round(recency_score(float(case["age_days"])), 6)
    return cases


if __name__ == "__main__":
    print("Song Age Clicks Impressions Hybrid Engagement Recency Final")
    for row in demo_score_table():
        print(
            f"{row['song']:>4} {row['age_days']:>3} {row['clicks']:>6} "
            f"{row['impressions']:>11} {row['hybrid_score']:>6.2f} "
            f"{row['engagement_score']:>10.4f} {row['recency_score']:>7.4f} "
            f"{row['final_score']:>7.4f}"
        )
