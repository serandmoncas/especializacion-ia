"""Rasgos de par asimétricos y logística con C elegido en validación."""

import numpy as np
from research_paths.evaluation import average_precision
from research_paths.prereq.features import fit_pair_logreg, pair_features, select_c


def _toy(rng, n_concepts=60, n_pairs=600):
    # Cada concepto tiene un «nivel» en la primera coordenada; b es prerrequisito de a si es de
    # nivel menor. Un modelo simétrico no puede aprenderlo; [a, b, a−b, a·b] sí.
    emb = rng.normal(size=(n_concepts, 8))
    level = emb[:, 0]
    i, j = rng.integers(0, n_concepts, (2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    return pair_features(emb[i], emb[j]), (level[j] < level[i]).astype(int)


def test_pair_features_are_asymmetric():
    a, b = np.array([[1.0, 2.0]]), np.array([[3.0, 5.0]])
    f = pair_features(a, b)
    assert f.shape == (1, 8)
    assert not np.allclose(f, pair_features(b, a))
    np.testing.assert_allclose(f[0, 4:6], [-2.0, -3.0])


def test_logreg_learns_direction_on_unseen_pairs():
    rng = np.random.default_rng(0)
    X_tr, y_tr = _toy(rng)
    X_val, y_val = _toy(rng)
    C = select_c(X_tr, y_tr, X_val, y_val)
    model = fit_pair_logreg(X_tr, y_tr, C)
    assert average_precision(y_val, model.predict_proba(X_val)[:, 1]) > 0.9
