"""Particiones: por autor (libro → temas), por orden del lector (recomendación) y libros nuevos."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def author_split(books: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """70/10/20 con todos los libros de un autor del mismo lado.

    Sin agrupar, un tomo de una serie en train y otro en test inflan las métricas: el modelo
    reconoce la serie, no el contenido.
    """
    ids, groups = books["book_id"].to_numpy(), books["main_author"].to_numpy()
    rest, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(ids, groups=groups)
    )
    tr, va = next(
        GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed).split(
            ids[rest], groups=groups[rest]
        )  # 0.125 · 0.8 = 0.1
    )
    return {"train": ids[rest][tr], "val": ids[rest][va], "test": ids[test]}


def new_books_holdout(book_ids: np.ndarray, frac: float = 0.1, seed: int = 42) -> np.ndarray:
    ids = np.asarray(book_ids)
    return np.random.default_rng(seed).choice(ids, size=round(frac * len(ids)), replace=False)


def user_split(ratings: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1):
    """Por lector: último `test_frac` (según `order`) a test; del resto, último `val_frac` a val.
    Cada lector conserva al menos una calificación en train."""
    r = ratings.sort_values(["user_id", "order"], kind="stable")
    size = r.groupby("user_id")["order"].transform("size")
    pos = r["order"]
    n_test = np.minimum(np.maximum(1, np.floor(size * test_frac)), size - 1).astype(int)
    rest = size - n_test
    n_val = np.minimum(np.floor(rest * val_frac), rest - 1).astype(int)
    is_test = pos >= size - n_test
    is_val = ~is_test & (pos >= rest - n_val)
    return r[~is_test & ~is_val], r[is_val], r[is_test]
