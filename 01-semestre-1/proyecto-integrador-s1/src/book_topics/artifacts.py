"""Entrena los modelos que usa la CLI y los guarda/carga con joblib.

Seguridad: joblib usa pickle, que puede ejecutar código al cargar. Este módulo solo carga archivos
generados en esta máquina por `entrenar`; nunca abrir un bundle descargado o recibido de terceros.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from book_topics.classify import TextTopicModel, apply_thresholds, book_text, fit_thresholds
from book_topics.data import load_books, load_descriptions, load_ratings, load_shelves
from book_topics.splits import author_split
from book_topics.taxonomy import TAU, Topic, label_books, load_taxonomy, topic_matrix


@dataclass
class Bundle:
    books: pd.DataFrame
    descriptions: pd.Series
    topics: list[Topic]
    labels: pd.DataFrame
    model: TextTopicModel
    thresholds: np.ndarray
    T: np.ndarray
    popularity: np.ndarray


def train_bundle(
    data_dir: Path,
    taxonomy_path: Path,
    *,
    min_df: int = 2,
    calibration_cv: int = 3,
    tau: float = TAU,
) -> Bundle:
    """Modelo de producción: entrena con train+val de la partición por autor; umbrales con val."""
    books = load_books(data_dir)
    desc = load_descriptions(data_dir)
    topics = load_taxonomy(taxonomy_path)
    labels = label_books(load_shelves(data_dir), topics, tau)
    has_text = desc.reindex(books.book_id).fillna("").str.len().to_numpy() > 0
    labeled = labels.reindex(books.book_id, fill_value=False).any(axis=1).to_numpy()
    parts = author_split(books[has_text & labeled], seed=42)
    texts = book_text(books.set_index("book_id")["title"], desc)
    val_ids = parts["val"] if len(parts["val"]) else parts["train"]
    fit_ids = np.r_[parts["train"], parts["val"]]
    model = TextTopicModel(min_df=min_df, calibration_cv=calibration_cv).fit(
        texts[fit_ids].tolist(), topic_matrix(labels, fit_ids).astype(bool)
    )
    thresholds = fit_thresholds(
        topic_matrix(labels, val_ids).astype(bool), model.predict_proba(texts[val_ids].tolist())
    )
    pop = (
        load_ratings(data_dir)
        .groupby("book_id")
        .size()
        .reindex(books.book_id, fill_value=0)
        .to_numpy(float)
    )
    return Bundle(
        books,
        desc,
        topics,
        labels,
        model,
        thresholds,
        topic_matrix(labels, books.book_id.to_numpy()),
        pop,
    )


def save_bundle(bundle: Bundle, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> Bundle:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No hay modelos en {path}. Ejecuta primero: python -m book_topics entrenar"
        )
    return joblib.load(path)  # solo archivos generados localmente por `entrenar` (ver docstring)


def predict_topics(bundle: Bundle, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    P = bundle.model.predict_proba(texts)
    return P, apply_thresholds(P, bundle.thresholds)
