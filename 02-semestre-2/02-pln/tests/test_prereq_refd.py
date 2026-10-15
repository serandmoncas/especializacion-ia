"""RefD (Liang et al., 2015) contra un ejemplo calculado a mano."""

import numpy as np
import pandas as pd
import pytest
from research_paths.prereq.refd import refd, refd_scores

LINKS = {"A": {"B", "C", "X"}, "B": set(), "C": {"B"}, "D": {"A", "B"}}
CONCEPTS = {"A", "B", "C", "D"}  # X no es un concepto del conjunto: se ignora


def test_refd_hand_example():
    # A enlaza a {B, C}; ambos «llegan» a B (B mismo y C → B): 2/2. B no enlaza a nada: 0.
    assert refd("A", "B", LINKS, CONCEPTS) == pytest.approx(1.0)
    assert refd("B", "A", LINKS, CONCEPTS) == pytest.approx(-1.0)
    # D enlaza a {A, B}: A llega a C y B no → 1/2. C enlaza a {B}, que no llega a D → 0.
    assert refd("D", "C", LINKS, CONCEPTS) == pytest.approx(0.5)


def test_refd_is_antisymmetric():
    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(8)]
    links = {n: {m for m in names if m != n and rng.random() < 0.4} for n in names}
    for a in names:
        for b in names:
            assert refd(a, b, links, set(names)) == pytest.approx(-refd(b, a, links, set(names)))


def test_refd_scores_over_dataframe():
    pairs = pd.DataFrame({"a_es": ["A", "B"], "b_es": ["B", "A"]})
    np.testing.assert_allclose(refd_scores(pairs, LINKS, CONCEPTS), [1.0, -1.0])
