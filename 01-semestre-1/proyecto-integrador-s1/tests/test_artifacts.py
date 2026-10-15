import pytest
from book_topics.artifacts import load_bundle, predict_topics, save_bundle, train_bundle


def test_train_save_load_roundtrip(medium_catalog, taxonomy_file, tmp_path):
    b = train_bundle(medium_catalog, taxonomy_file, min_df=1, calibration_cv=2)
    assert b.T.shape == (54, 4) and len(b.thresholds) == 4 and len(b.popularity) == 54
    save_bundle(b, tmp_path / "m.joblib")
    b2 = load_bundle(tmp_path / "m.joblib")
    assert b2.books.title.tolist() == b.books.title.tolist()
    P, assigned = predict_topics(b2, ["a dragon wizard on a quest"])
    assert P.shape == (1, 4) and assigned[0, 0]


def test_load_missing_bundle_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="entrenar"):
        load_bundle(tmp_path / "no.joblib")
