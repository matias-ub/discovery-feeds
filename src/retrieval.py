"""Candidate retrieval for a discovery feed."""

from typing import List

import pandas as pd

from .embeddings import generate_embeddings
from .vector_index import VectorIndex


def retrieve_candidates(
    query: str,
    index: VectorIndex,
    articles: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """Retrieve top-k similar articles for a query."""
    query_embeddings = generate_embeddings([query])
    scores, indices = index.search(query_embeddings, top_k)
    idx_list: List[int] = indices[0].tolist()
    sim_list: List[float] = scores[0].tolist()

    candidates = articles.iloc[idx_list].copy()
    candidates["similarity"] = sim_list
    return candidates.reset_index(drop=True)
