"""Dataset loader for AG News samples used in the recommender."""

from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
import pandas as pd
from datasets import load_dataset

LABEL_MAP: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def load_ag_news_sample(per_label: int = 500, seed: int = 42) -> pd.DataFrame:
    """Load a balanced AG News sample with recency and popularity signals."""
    dataset = load_dataset("ag_news", split="train")
    df = dataset.to_pandas().groupby("label", as_index=False).head(per_label)

    df["category"] = df["label"].map(LABEL_MAP)
    df["title"] = df["text"].str.split(".").str[0]
    df["description"] = df["text"]

    rng = np.random.default_rng(seed)
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    days_ago = rng.integers(0, 365, size=len(df))
    published = [base - timedelta(days=int(d)) for d in days_ago]
    df["published_at"] = [dt.isoformat().replace("+00:00", "Z") for dt in published]

    df["popularity"] = rng.uniform(0.1, 0.95, size=len(df)).round(4)

    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    return df[["id", "title", "description", "category", "popularity", "published_at"]]
