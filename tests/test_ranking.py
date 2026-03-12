import pandas as pd

from src.ranking import rank_candidates


def test_rank_candidates_sorted_descending() -> None:
    candidates = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "title": ["a", "b", "c"],
            "description": ["x", "y", "z"],
            "topic": ["ml", "ml", "ml"],
            "similarity": [0.9, 0.6, 0.3],
            "popularity": [0.2, 0.9, 0.1],
            "published_at": [
                "2025-01-10T00:00:00Z",
                "2024-12-01T00:00:00Z",
                "2024-11-01T00:00:00Z",
            ],
        }
    )

    ranked = rank_candidates(candidates, top_k=3)
    scores = ranked["score"].to_list()
    assert scores == sorted(scores, reverse=True)
