import numpy as np
import pytest
from applied_recsys.baselines import BiasModel, popularity_scores
from applied_recsys.data import random_split, temporal_split, to_csr
from applied_recsys.neighbors import ItemKNN, item_similarity, keep_top_k
from applied_recsys.synthetic import synthetic_ratings
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.metrics.pairwise import cosine_similarity


@pytest.fixture(scope="module")
def data():
    df, truth = synthetic_ratings(n_users=300, n_items=200, density=0.1, seed=0)
    return df, truth, (300, 200)


def test_bias_model_reaches_joint_ridge_optimum(data):
    df, _, shape = data
    R = to_csr(df, shape)
    lu, li = 5.0, 15.0
    model = BiasModel(reg_user=lu, reg_item=li, n_iter=200).fit(R)
    # Mismo problema como un único ridge con variables indicadoras, resuelto con lsqr.
    n = len(df)
    A = sparse.hstack(
        [
            sparse.csr_matrix((np.ones(n), (np.arange(n), df.u)), shape=(n, shape[0])),
            sparse.csr_matrix((np.ones(n), (np.arange(n), df.i)), shape=(n, shape[1])),
        ]
    ).tocsr()
    penalty = sparse.diags(np.r_[np.full(shape[0], np.sqrt(lu)), np.full(shape[1], np.sqrt(li))])
    A_aug = sparse.vstack([A, penalty])
    y_aug = np.r_[df.rating.to_numpy() - model.mu, np.zeros(sum(shape))]
    sol = lsqr(A_aug, y_aug, atol=1e-12, btol=1e-12, iter_lim=10_000)[0]
    np.testing.assert_allclose(model.bu, sol[: shape[0]], atol=1e-6)
    np.testing.assert_allclose(model.bi, sol[shape[0] :], atol=1e-6)


def test_bias_model_beats_global_mean(data):
    df, _, shape = data
    tr, te = random_split(df, 0.2, seed=1)
    model = BiasModel().fit(to_csr(tr, shape))
    err_bias = np.sqrt(np.mean((te.rating - model.predict(te.u, te.i)) ** 2))
    err_mean = np.sqrt(np.mean((te.rating - tr.rating.mean()) ** 2))
    assert err_bias < err_mean


def test_item_similarity_matches_sklearn_and_shrink_lowers_it(data):
    df, _, shape = data
    X = to_csr(df, shape)
    sim = item_similarity(X)
    ref = cosine_similarity(X.T)
    np.fill_diagonal(ref, 0)
    np.testing.assert_allclose(sim, ref, atol=1e-10)
    shrunk = item_similarity(X, shrink=10)
    assert np.all(np.abs(shrunk) <= np.abs(sim) + 1e-12)


def test_keep_top_k():
    sim = np.array([[0, 0.9, 0.1], [0.9, 0, 0.5], [0.1, 0.5, 0]])
    top = keep_top_k(sim, 1)
    assert np.count_nonzero(top, axis=0).tolist() == [1, 1, 1]
    assert top[1, 0] == 0.9 and top[1, 2] == 0.5


def test_item_knn_improves_on_biases(data):
    df, _, shape = data
    tr, te = random_split(df, 0.2, seed=2)
    R = to_csr(tr, shape)
    bias = BiasModel().fit(R)
    knn = ItemKNN(k=30, shrink=10).fit(R, bias)
    err_knn = np.sqrt(np.mean((te.rating - knn.predict(te.u.to_numpy(), te.i.to_numpy())) ** 2))
    err_bias = np.sqrt(np.mean((te.rating - bias.predict(te.u, te.i)) ** 2))
    assert err_knn < err_bias


def test_popularity_counts(data):
    df, _, shape = data
    pop = popularity_scores(to_csr(df, shape))
    np.testing.assert_array_equal(pop, np.bincount(df.i, minlength=shape[1]))


def test_temporal_split_respects_time_per_user():
    df, _ = synthetic_ratings(n_users=100, n_items=80, density=0.2, seed=3)
    tr, te = temporal_split(df, 0.25)
    last_train = tr.groupby("u").ts.max()
    first_test = te.groupby("u").ts.min()
    common = first_test.index.intersection(last_train.index)
    assert (first_test[common] >= last_train[common]).all()
    assert set(te.i) <= set(tr.i)


def test_denominator_regularization_matters_on_sparse_data(data):
    # Regresión: sin λ en el denominador, kNN amplificaba ruido y quedaba peor que los sesgos.
    df, _, shape = data
    tr, te = random_split(df, 0.2, seed=2)
    R = to_csr(tr, shape)
    bias = BiasModel().fit(R)
    u, i, y = te.u.to_numpy(), te.i.to_numpy(), te.rating.to_numpy()

    def err(model):
        return np.sqrt(np.mean((y - model.predict(u, i)) ** 2))

    raw = ItemKNN(k=30, shrink=10, denom_reg=0.0).fit(R, bias)
    reg = ItemKNN(k=30, shrink=10, denom_reg=0.5).fit(R, bias)
    assert err(reg) < err(bias) < err(raw)
