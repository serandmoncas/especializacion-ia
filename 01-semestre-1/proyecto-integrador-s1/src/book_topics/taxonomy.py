"""De estantes ruidosos a temas: concentración, taxonomía curada y etiquetado por libro."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Barrido sobre goodbooks-10k con temas.yaml (40 temas): τ = 0.05 → mediana 3 temas/libro, 0 libros
# sin temas, tema mínimo 100 libros; τ = 0.10 → 3, 0, 53; τ = 0.15 → 2, 0, 28; τ = 0.20 → 2, 12, 9.
# Regla: el menor τ con mediana entre 2 y 4 y todos los temas con ≥ 50 libros.
TAU = 0.05


def gini(x: np.ndarray) -> float:
    """Coeficiente de Gini de valores no negativos: 0 = repartido parejo; → 1 = concentrado."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def shelf_concentration(shelves: pd.DataFrame, min_books: int = 100) -> pd.DataFrame:
    """Dos medidas por estante para separar temas de organización personal.

    - `gini`: Gini, sobre todos los libros, de la proporción del estante en los estantes del libro.
      Era el criterio de la spec, pero en goodbooks-10k se confunde con la rareza (Spearman con
      `n_books` ≈ −0.97): fantasy (0.86) cae en la misma zona que abandoned (0.80). Se conserva solo
      para documentar por qué no sirve.
    - `corr_popularidad`: correlación, sobre todos los libros, entre log(1 + conteo del estante) y
      log(1 + total de estantes del libro). Un estante organizativo (owned, to-read) crece con
      cuánta gente estantea el libro; uno temático depende del contenido. Es el criterio usado.

    Ordenado por `corr_popularidad` ascendente (los más temáticos primero).
    """
    totals = shelves.groupby("book_id")["count"].sum()
    books = totals.index
    log_total = np.log1p(totals.to_numpy(float))
    share = shelves.assign(share=shelves["count"] / shelves["book_id"].map(totals))
    rows = []
    for shelf, g in share.groupby("shelf"):
        n = g["book_id"].nunique()
        if n < min_books:
            continue
        counts = g.set_index("book_id")["count"].reindex(books, fill_value=0).to_numpy(float)
        shares = np.r_[g["share"].to_numpy(), np.zeros(len(books) - len(g))]
        log_counts = np.log1p(counts)
        corr = float(np.corrcoef(log_counts, log_total)[0, 1]) if log_counts.std() > 0 else 0.0
        rows.append((shelf, n, gini(shares), corr))
    return pd.DataFrame(rows, columns=["shelf", "n_books", "gini", "corr_popularidad"]).sort_values(
        "corr_popularidad", ignore_index=True
    )


@dataclass(frozen=True)
class Topic:
    key: str
    nombre: str
    estantes: tuple[str, ...]


def load_taxonomy(path: str | Path) -> list[Topic]:
    raw = yaml.safe_load(Path(path).read_text())
    topics, seen = [], {}
    for key, spec in raw.items():
        shelves = tuple(spec.get("estantes") or ())
        if not shelves:
            raise ValueError(f"El tema {key!r} no tiene estantes")
        for s in shelves:
            if s in seen:
                raise ValueError(f"El estante {s!r} está en {seen[s]!r} y en {key!r}")
            seen[s] = key
        topics.append(Topic(key, spec["nombre"], shelves))
    return topics


def label_books(shelves: pd.DataFrame, topics: list[Topic], tau: float = TAU) -> pd.DataFrame:
    """Un tema se asigna si sus estantes suman ≥ τ del total de conteos *temáticos* del libro."""
    to_topic = {s: t.key for t in topics for s in t.estantes}
    topical = shelves[shelves["shelf"].isin(to_topic)].assign(
        topic=lambda d: d["shelf"].map(to_topic)
    )
    by_topic = topical.pivot_table(
        index="book_id", columns="topic", values="count", aggfunc="sum", fill_value=0
    ).reindex(columns=[t.key for t in topics], fill_value=0)
    share = by_topic.div(by_topic.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    all_books = pd.Index(sorted(shelves["book_id"].unique()), name="book_id")
    return (share >= tau).reindex(all_books, fill_value=False)


def topic_matrix(labels: pd.DataFrame, book_ids: np.ndarray) -> np.ndarray:
    return labels.reindex(book_ids, fill_value=False).to_numpy(dtype=float)
