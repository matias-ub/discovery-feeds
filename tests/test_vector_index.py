import numpy as np

from src.vector_index import VectorIndex


def test_faiss_index_returns_k_results() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(10, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = VectorIndex.build(embeddings)
    query = embeddings[:1]
    scores, indices = index.search(query, top_k=5)

    assert scores.shape == (1, 5)
    assert indices.shape == (1, 5)
