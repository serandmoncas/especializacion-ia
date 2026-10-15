"""Ejercicio 4: TF-IDF desde cero, igual a sklearn (idf suavizado, norma L2). Ver README.md."""

import re

import numpy as np

TOKEN = re.compile(r"(?u)\b\w+\b")


def tfidf(docs: list[str]) -> tuple[list[str], np.ndarray]:
    raise NotImplementedError


def check() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = ["El aprendizaje por refuerzo", "aprendizaje profundo y refuerzo", "el cacao", ""]
    vocab, m = tfidf(docs)
    ref = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    r = ref.fit_transform(docs).toarray()
    assert vocab == list(ref.get_feature_names_out())
    np.testing.assert_allclose(m, r, atol=1e-12)
    print("ej04 ok — igual que TfidfVectorizer")


if __name__ == "__main__":
    check()
