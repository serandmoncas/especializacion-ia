import numpy as np
import pandas as pd
from book_topics.recommend import (
    center_ratings,
    explain,
    hybrid_scores,
    profile_from_titles,
    topic_scores,
    user_topic_profiles,
    zscore_rows,
)
from scipy import sparse

T = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]], float)


def test_center_ratings():
    R = center_ratings(pd.DataFrame({"u": [0, 0, 1], "i": [0, 1, 2], "rating": [5, 3, 4]}), 2, 5)
    assert R[0, 0] == 1 and R[0, 1] == -1 and R[1, 2] == 0


def test_profile_weights_topics_by_centered_rating():
    p = user_topic_profiles(sparse.csr_matrix(np.array([[1.0, 0, -1.0, 0, 0]])), T)[0]
    assert p[0] > 0 and p[2] < 0


def test_topic_scores_cosine_and_null_profile():
    S = topic_scores(np.array([[1.0, 0, 0], [0, 0, 0]]), T)
    assert S[0].argmax() == 0 and np.all(S[1] == 0) and not np.isnan(S).any() and S[0, 4] == 0


def test_profile_from_titles_cancels_to_zero():
    assert np.allclose(profile_from_titles([0], [0], T), 0)


def test_hybrid_and_zscore():
    a, b = np.array([[1.0, 2, 3]]), np.array([[3.0, 2, 1]])
    assert abs(zscore_rows(a).mean()) < 1e-12
    assert np.allclose(hybrid_scores(a, b, 1.0), zscore_rows(a))
    assert np.allclose(hybrid_scores(a, b, 0.5), 0)


def test_explain_picks_largest_contributions():
    names = ["fantasía", "aventura", "romance"]
    assert explain(np.array([0.9, 0.5, 0.1]), np.ones(3), names, top=2) == ["fantasía", "aventura"]
    assert explain(np.array([0.9, 0.5, 0.1]), np.zeros(3), names) == []
