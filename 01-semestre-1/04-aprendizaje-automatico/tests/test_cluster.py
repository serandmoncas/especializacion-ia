import numpy as np
import pytest
from applied_ml import KMeans
from applied_ml.cluster import kmeans_plusplus
from sklearn import cluster
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def test_recovers_well_separated_blobs():
    X, y = make_blobs(600, centers=4, cluster_std=0.6, random_state=0)
    km = KMeans(4, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) == pytest.approx(1.0)


def test_inertia_matches_sklearn():
    X, _ = make_blobs(1000, centers=6, cluster_std=1.5, random_state=1)
    ours = KMeans(6, n_init=10, random_state=0).fit(X).inertia_
    ref = cluster.KMeans(6, n_init=10, random_state=0).fit(X).inertia_
    assert ours == pytest.approx(ref, rel=0.01)


def test_plusplus_beats_random_init_on_average():
    X, _ = make_blobs(800, centers=10, cluster_std=0.5, random_state=2)
    pp = [KMeans(10, init="k-means++", n_init=1, random_state=s).fit(X).inertia_ for s in range(15)]
    rnd = [KMeans(10, init="random", n_init=1, random_state=s).fit(X).inertia_ for s in range(15)]
    assert np.mean(pp) < np.mean(rnd)


def test_plusplus_picks_distinct_points():
    X, _ = make_blobs(200, centers=3, random_state=3)
    C = kmeans_plusplus(X, 5, np.random.default_rng(0))
    assert len(np.unique(C, axis=0)) == 5


def test_predict_assigns_to_nearest_center():
    X, _ = make_blobs(300, centers=3, random_state=4)
    km = KMeans(3, random_state=0).fit(X)
    np.testing.assert_array_equal(km.predict(X), km.labels_)
