import numpy as np

from src import embeddings


def test_generate_embeddings_shape(monkeypatch) -> None:
    class DummyModel:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return np.zeros((len(texts), 8), dtype=np.float32)

    monkeypatch.setattr(embeddings, "_load_model", lambda _: DummyModel())
    texts = ["hello world", "machine learning"]
    vectors = embeddings.generate_embeddings(texts)
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(texts)
    assert vectors.ndim == 2
