"""Validación de las etiquetas contra los *subjects* de Open Library (fuente independiente)."""

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

SUBJECT_KEYWORDS = {
    "fantasia": ["fantasy"],
    "ciencia_ficcion": ["science fiction"],
    "romance": ["romance", "love stories"],
    "misterio": ["mystery", "detective"],
    "historia": ["history"],
    "biografia": ["biography", "memoir", "autobiography"],
    "autoayuda": ["self-help", "self-actualization"],
    "terror": ["horror"],
    "poesia": ["poetry"],
    "juvenil": ["young adult", "juvenile fiction"],
}


def _fetch_one(isbn: str) -> list[str]:
    url = "https://openlibrary.org/search.json?" + urllib.parse.urlencode(
        {"q": f"isbn:{isbn}", "fields": "subject", "limit": 1}
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        docs = json.load(resp).get("docs", [])
    return docs[0].get("subject", []) if docs else []


def fetch_subjects(
    isbns: list[str],
    cache_path: Path,
    pause: float = 1.0,
    fetch: Callable[[str], list[str]] | None = None,
) -> dict[str, list[str]]:
    """Consulta con caché en disco y pausa entre llamadas (Open Library pide uso moderado).
    Un error de red deja ese ISBN como `None` en la caché: se excluye, no se reintenta en bucle."""
    fetch = fetch or _fetch_one
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for isbn in isbns:
        if isbn in cache:
            continue
        try:
            cache[isbn] = fetch(isbn)
        except OSError:
            cache[isbn] = None
        cache_path.write_text(json.dumps(cache))
        time.sleep(pause)
    return {i: cache[i] for i in isbns if cache.get(i) is not None}


def subjects_to_topics(subjects: list[str]) -> set[str]:
    low = [s.lower() for s in subjects]
    return {t for t, kws in SUBJECT_KEYWORDS.items() if any(k in s for s in low for k in kws)}


def agreement(labels: pd.DataFrame, ol_topics: dict[int, set[str]]) -> pd.DataFrame:
    books = [b for b in ol_topics if b in labels.index]
    rows = []
    for t in SUBJECT_KEYWORDS:
        if t not in labels.columns:
            continue
        ours = labels.loc[books, t].astype(bool).to_numpy()
        theirs = [t in ol_topics[b] for b in books]
        same = len(set(ours)) == 1 and len(set(theirs)) == 1 and ours[0] == theirs[0]
        kappa = 1.0 if same else float(cohen_kappa_score(ours, theirs))
        rows.append((t, kappa, int(ours.sum()), int(sum(theirs))))
    return pd.DataFrame(rows, columns=["topic", "kappa", "n_ours", "n_ol"])
