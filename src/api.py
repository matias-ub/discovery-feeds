"""FastAPI service exposing recommendations."""

from pathlib import Path

from fastapi import FastAPI, Query

from .recommender import Recommender

app = FastAPI(title="Discovery Feed Recommender")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARTICLES_PATH = DATA_DIR / "articles.csv"
INDEX_PATH = DATA_DIR / "index.faiss"

recommender = Recommender(str(ARTICLES_PATH), str(INDEX_PATH))


@app.get("/recommend")
def recommend(
    q: str = Query(..., min_length=2, description="User query"),
    k: int = Query(10, ge=1, le=20, description="Number of results"),
):
    results = recommender.recommend(q, top_k=k)
    return {"query": q, "results": results}
