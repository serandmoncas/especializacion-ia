"""Carga de MovieLens, índices contiguos, particiones y matrices dispersas."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def load_movielens_1m(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(calificaciones, películas). Calificaciones con columnas user, item, rating, ts y los
    índices contiguos `u`, `i` (0…n−1) que usan las matrices."""
    root = Path(data_dir) / "ml-1m"
    ratings = pd.read_csv(
        root / "ratings.dat", sep="::", engine="python", names=["user", "item", "rating", "ts"]
    )
    movies = pd.read_csv(
        root / "movies.dat",
        sep="::",
        engine="python",
        names=["item", "title", "genres"],
        encoding="latin-1",
    )
    ratings["u"] = ratings["user"].astype("category").cat.codes.astype(np.int32)
    ratings["i"] = ratings["item"].astype("category").cat.codes.astype(np.int32)
    item_map = ratings.drop_duplicates("i").set_index("item")["i"]
    movies = movies[movies["item"].isin(item_map.index)].assign(i=lambda m: m["item"].map(item_map))
    return ratings, movies.sort_values("i").reset_index(drop=True)


def temporal_split(
    ratings: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Por usuario: el `test_frac` más reciente de sus calificaciones va a test.

    Simula recomendar lo próximo que la persona verá, sin usar su futuro. Se descartan del test los
    ítems que no aparecen en train (ítems fríos): ningún modelo colaborativo puede puntuarlos.
    """
    r = ratings.sort_values(["u", "ts", "i"], kind="stable")
    rank = r.groupby("u").cumcount()
    size = r.groupby("u")["u"].transform("size")
    n_test = np.maximum(1, np.floor(size * test_frac)).astype(int)
    is_test = rank >= size - n_test
    train, test = r[~is_test], r[is_test]
    test = test[test["i"].isin(train["i"].unique())]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def random_split(
    ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición aleatoria de calificaciones (la clásica del premio Netflix). Más fácil que la
    temporal: el modelo ve calificaciones posteriores del mismo usuario."""
    mask = np.random.default_rng(seed).random(len(ratings)) < test_frac
    train, test = ratings[~mask], ratings[mask]
    test = test[test["i"].isin(train["i"].unique()) & test["u"].isin(train["u"].unique())]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def to_csr(
    df: pd.DataFrame, shape: tuple[int, int], values: str | None = "rating"
) -> sparse.csr_matrix:
    """Matriz usuarios × ítems. values=None → 1 por cada interacción (implícita)."""
    data = np.ones(len(df)) if values is None else df[values].to_numpy(dtype=float)
    return sparse.csr_matrix((data, (df["u"].to_numpy(), df["i"].to_numpy())), shape=shape)
