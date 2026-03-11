"""Ranking utilities combining similarity, popularity, and recency."""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd


def _recency_score(published_at: str, now: Optional[datetime] = None) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    days_ago = max((now - published_dt).days, 0)
    return 1.0 / (1.0 + days_ago)


def rank_candidates(candidates: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Rank candidates using a weighted score."""
    now = datetime.now(timezone.utc)
    scores = []
    for _, row in candidates.iterrows():
        similarity = float(row.get("similarity", 0.0))
        popularity = float(row.get("popularity", 0.0))
        popularity = max(min(popularity, 1.0), 0.0)
        recency = _recency_score(str(row.get("published_at", "1970-01-01")), now)
        score = 0.7 * similarity + 0.2 * popularity + 0.1 * recency
        scores.append(score)

    ranked = candidates.copy()
    ranked["score"] = scores
    ranked = ranked.sort_values("score", ascending=False).head(top_k)
    return ranked.reset_index(drop=True)
