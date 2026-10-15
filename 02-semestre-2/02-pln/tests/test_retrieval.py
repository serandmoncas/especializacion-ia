"""BM25 contra un cálculo a mano; codificador de hashing y coseno."""

import math

import numpy as np
from research_paths.retrieval import BM25, cosine_scores, hashing_encoder, simple_tokenize


def test_simple_tokenize_keeps_accents_and_lowercases():
    assert simple_tokenize("Información, ¡Aprendizaje!") == ["información", "aprendizaje"]


def test_bm25_matches_hand_computation():
    docs = [["a", "b"], ["a"], ["c", "c", "a"]]
    bm = BM25().fit(docs)
    # N = 3, n(c) = 1, avgdl = 2, |d3| = 3, f = 2
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2))
    np.testing.assert_allclose(bm.scores(["c"]), [0, 0, expected])


def test_bm25_ranks_the_matching_document_first_and_ignores_unknown_terms():
    docs = [simple_tokenize(t) for t in ("el gato negro", "el perro", "aprendizaje por refuerzo")]
    s = BM25().fit(docs).scores(simple_tokenize("refuerzo xyz"))
    assert s.argmax() == 2 and s[0] == s[1] == 0


def test_bm25_common_term_has_small_positive_idf():
    bm = BM25().fit([["a"], ["a", "b"]])
    assert 0 < bm.idf["a"] < bm.idf["b"]


def test_hashing_encoder_is_normalized_and_deterministic():
    enc = hashing_encoder(64)
    a = enc(["aprendizaje por refuerzo", ""], "query")
    np.testing.assert_allclose(np.linalg.norm(a[0]), 1)
    assert np.all(a[1] == 0)
    np.testing.assert_array_equal(a, enc(["aprendizaje por refuerzo", ""], "passage"))


def test_cosine_scores_rank_similar_text_first():
    enc = hashing_encoder(256)
    docs = enc(["gato negro", "aprendizaje por refuerzo profundo", "perro"], "passage")
    q = enc(["refuerzo profundo"], "query")[0]
    assert cosine_scores(q, docs).argmax() == 1


def test_cosine_scores_handles_zero_vectors():
    assert np.all(cosine_scores(np.zeros(3), np.zeros((2, 3))) == 0)
