"""Recuperación: BM25 desde cero, codificadores de texto y orden denso por coseno."""

import math
import re
import zlib
from collections import Counter
from collections.abc import Callable

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)

# encode(textos, kind) -> matriz (n, d) con filas de norma 1 (o 0 si el texto está vacío);
# kind ∈ {"query", "passage"} (e5 usa prefijos distintos).
Encoder = Callable[[list[str], str], np.ndarray]


def simple_tokenize(text: str) -> list[str]:
    """Minúsculas y palabras Unicode (conserva tildes)."""
    return _WORD.findall(text.lower())


class BM25:
    """Okapi BM25 con idf = ln(1 + (N − n + 0,5)/(n + 0,5)), siempre positivo (como Lucene)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.tf = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        mean = float(self.lengths.mean()) if len(docs) else 0.0
        self.avgdl = mean if mean > 0 else 1.0
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        return self

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            f = np.array([tf[term] for tf in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def hashing_encoder(dim: int = 256) -> Encoder:
    """Bolsa de palabras con hashing (sin modelo): para tests y notebooks; ignora `kind`."""

    def encode(texts: list[str], kind: str = "query") -> np.ndarray:
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            for word in simple_tokenize(text):
                out[i, zlib.crc32(word.encode()) % dim] += 1.0
        return _normalize_rows(out)

    return encode


def e5_encoder(
    model_name: str = "intfloat/multilingual-e5-base",
    device: str | None = None,
    batch_size: int = 32,
) -> Encoder:
    """e5 multilingüe vía sentence-transformers (se descarga la primera vez, ~1,1 GB)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str], kind: str = "query") -> np.ndarray:
        prefix = "query: " if kind == "query" else "passage: "
        return model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return encode


def cosine_scores(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    """Coseno entre un vector y cada fila; los vectores nulos dan 0."""
    q = np.asarray(query_vec, dtype=float)
    q = q / (np.linalg.norm(q) or 1.0)
    return _normalize_rows(np.asarray(doc_mat, dtype=float)) @ q
