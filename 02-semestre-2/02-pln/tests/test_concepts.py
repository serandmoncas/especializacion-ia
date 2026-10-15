"""Tema → conceptos candidatos → matriz de prerrequisitos → ruta."""

import numpy as np
import pytest
from research_paths.concepts import (
    candidate_concepts,
    concept_text,
    order_route,
    prereq_matrix,
    prerequisite_of,
    resolve_topic,
)
from research_paths.retrieval import hashing_encoder


def test_concept_text():
    assert (
        concept_text("Null_hypothesis", "La hipótesis nula…")
        == "Null hypothesis. La hipótesis nula…"
    )
    assert concept_text("Media", "") == "Media"


def test_resolve_topic_suggests_on_disambiguation(wiki_world):
    assert resolve_topic("Aprendizaje por refuerzo", wiki_world) == ("Aprendizaje por refuerzo", [])
    title, suggestions = resolve_topic("Mercurio", wiki_world)
    assert title is None and "Mercurio (planeta)" in suggestions


def test_candidate_concepts_skip_pages_without_text_and_prefer_related(wiki_world):
    out = candidate_concepts("Aprendizaje por refuerzo", wiki_world, hashing_encoder(), k=3)
    assert "Estados Unidos" not in out  # sin extracto
    assert len(out) == 3 and "Aprendizaje por refuerzo" not in out


def test_order_route_puts_a_chain_in_order():
    concepts = ["C", "A", "B"]  # A → B → C (A es prerrequisito de B, B de C)
    P = np.zeros((3, 3))
    P[0, 2] = P[0, 1] = 0.9  # C necesita B y A
    P[2, 1] = 0.9  # B necesita A
    assert order_route(concepts, P) == ["A", "B", "C"]


def test_order_route_returns_a_permutation_even_with_cycles():
    rng = np.random.default_rng(0)
    concepts = [f"c{i}" for i in range(6)]
    P = rng.random((6, 6))
    assert sorted(order_route(concepts, P)) == sorted(concepts)


def test_prerequisite_of():
    P = np.array([[0, 0.9], [0.1, 0]])
    assert prerequisite_of(["A", "B"], P) == {"A": [], "B": ["A"]}


def test_prereq_matrix_shape_and_zero_diagonal():
    pytest.importorskip("transformers")
    from research_paths.textclf import tiny_model

    concepts = ["Probabilidad", "Recompensa", "Markov"]
    texts = {c: f"{c} texto" for c in concepts}
    model, tok = tiny_model(list(texts.values()))
    P = prereq_matrix(concepts, texts, model, tok, max_length=64)
    assert P.shape == (3, 3) and np.all(np.diag(P) == 0) and np.all((P >= 0) & (P <= 1))
