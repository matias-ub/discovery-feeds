"""Vector index utilities built on FAISS."""

from typing import Tuple

import faiss
import numpy as np


class VectorIndex:
    """Wrapper around a FAISS index for cosine similarity search."""

    def __init__(self, index: faiss.Index) -> None:
        self.index = index

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "VectorIndex":
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(index)

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        return cls(faiss.read_index(path))

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices
