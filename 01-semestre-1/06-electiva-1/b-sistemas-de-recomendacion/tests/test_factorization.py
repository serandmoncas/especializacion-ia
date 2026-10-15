import numpy as np
import pytest
from applied_recsys.baselines import BiasModel
from applied_recsys.data import random_split, to_csr
from applied_recsys.factorization import ExplicitALS, FunkSVD, ImplicitALS, PureSVD, ridge_rows
from applied_recsys.synthetic import synthetic_ratings
from scipy import sparse


@pytest.fixture(scope="module")
def split():
    df, _ = synthetic_ratings(n_users=400, n_items=300, density=0.1, noise=0.3, mnar=0.0, seed=0)
    tr, te = random_split(df, 0.2, seed=0)
    return tr, te, (400, 300)


def test_ridge_rows_is_exact_ridge_solution():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(10, 3))
    target = sparse.csr_matrix(np.where(rng.random((4, 10)) < 0.5, rng.normal(size=(4, 10)), 0))
    out = ridge_rows(target, F, reg=0.2)
    for row in range(4):
        cols = target[row].indices
        if len(cols) == 0:
            continue
        A = np.vstack([F[cols], np.sqrt(0.2 * len(cols)) * np.eye(3)])
        b = np.r_[target[row].data, np.zeros(3)]
        np.testing.assert_allclose(out[row], np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10)


def test_als_loss_is_monotone(split):
    tr, _, shape = split
    R = to_csr(tr, shape)
    als = ExplicitALS(k=8, reg=0.05, n_iter=8).fit(R, BiasModel().fit(R))
    assert np.all(np.diff(als.loss_history) <= 1e-9)


@pytest.mark.parametrize("model_name", ["als", "sgd"])
def test_factorization_beats_biases_on_low_rank_data(split, model_name):
    tr, te, shape = split
    R = to_csr(tr, shape)
    bias = BiasModel().fit(R)
    model = (
        ExplicitALS(k=8, reg=0.05, n_iter=10).fit(R, bias)
        if model_name == "als"
        else FunkSVD(k=8, lr=0.02, reg=0.05, epochs=40, batch=256).fit(R)
    )
    u, i, y = te.u.to_numpy(), te.i.to_numpy(), te.rating.to_numpy()
    err_model = np.sqrt(np.mean((y - model.predict(u, i)) ** 2))
    err_bias = np.sqrt(np.mean((y - bias.predict(u, i)) ** 2))
    assert err_model < 0.95 * err_bias


def test_implicit_als_trick_matches_dense_solution():
    rng = np.random.default_rng(1)
    R = sparse.random(6, 9, density=0.4, random_state=2, format="csr") * 5
    Y = rng.normal(size=(9, 3))
    alpha, reg = 2.0, 0.5
    C = R.copy()
    C.data = 1 + alpha * C.data
    fast = ImplicitALS._solve(C, Y, reg)
    dense_c = 1 + alpha * R.toarray()
    pref = (R.toarray() > 0).astype(float)
    for u in range(6):
        if R[u].nnz == 0:
            continue
        Cu = np.diag(dense_c[u])
        A = Y.T @ Cu @ Y + reg * np.eye(3)
        np.testing.assert_allclose(fast[u], np.linalg.solve(A, Y.T @ Cu @ pref[u]), atol=1e-10)


def test_pure_svd_equals_dense_projection():
    R = sparse.random(30, 20, density=0.3, random_state=0, format="csr")
    model = PureSVD(k=5).fit(R)
    _, _, vt = np.linalg.svd(R.toarray(), full_matrices=False)
    V = vt[:5].T
    np.testing.assert_allclose(model.scores(np.arange(30)), R.toarray() @ V @ V.T, atol=1e-8)
