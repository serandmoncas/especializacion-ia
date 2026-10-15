"""Ejercicio 4: TF-IDF desde cero, igual a sklearn (idf suavizado, norma L2). Ver README.md."""

import re

import numpy as np

TOKEN = re.compile(r"(?u)\b\w+\b")


def tfidf(docs: list[str]) -> tuple[list[str], np.ndarray]:
    tokenized = [TOKEN.findall(d.lower()) for d in docs]
    vocab = sorted({t for doc in tokenized for t in doc})
    index = {t: i for i, t in enumerate(vocab)}
    tf = np.zeros((len(docs), len(vocab)))
    for r, doc in enumerate(tokenized):
        for t in doc:
            tf[r, index[t]] += 1
    df = (tf > 0).sum(0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    m = tf * idf
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return vocab, m / np.where(norms == 0, 1, norms)


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
