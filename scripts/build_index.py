"""Build and save the FAISS index for article embeddings."""

from pathlib import Path

import pandas as pd

from src.embeddings import generate_embeddings
from src.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    articles = pd.read_csv(DATA_DIR / "articles.csv")
    texts = (
        articles["title"].fillna("")
        + ". "
        + articles["description"].fillna("")
    ).tolist()
    embeddings = generate_embeddings(texts)
    index = VectorIndex.build(embeddings)
    index_path = DATA_DIR / "index.faiss"
    index.save(str(index_path))
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()
