import numpy as np

from src.embeddings import generate_embeddings


def test_generate_embeddings_shape() -> None:
    texts = ["hello world", "machine learning"]
    embeddings = generate_embeddings(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.ndim == 2
