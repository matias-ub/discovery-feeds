"""End-to-end recommender orchestration."""

from typing import Dict, List

import pandas as pd

from .ranking import rank_candidates
from .retrieval import retrieve_candidates
from .vector_index import VectorIndex


class Recommender:
    """High-level recommender that loads data and a vector index."""

    def __init__(self, articles_path: str, index_path: str) -> None:
        self.articles = pd.read_csv(articles_path)
        self.index = VectorIndex.load(index_path)

    def recommend(self, query: str, top_k: int = 10) -> List[Dict]:
        candidates = retrieve_candidates(query, self.index, self.articles, top_k=20)
        ranked = rank_candidates(candidates, top_k=top_k)
        fields = [
            "id",
            "title",
            "description",
            "topic",
            "popularity",
            "published_at",
            "similarity",
            "score",
        ]
        return ranked[fields].to_dict(orient="records")
