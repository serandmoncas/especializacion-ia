import pytest
from book_topics.artifacts import load_bundle, predict_topics, save_bundle, train_bundle


def test_train_save_load_roundtrip(medium_catalog, taxonomy_file, tmp_path):
    b = train_bundle(medium_catalog, taxonomy_file, min_df=1, calibration_cv=2, min_words=1)
    assert b.T.shape == (54, 4) and len(b.thresholds) == 4 and len(b.popularity) == 54
    save_bundle(b, tmp_path / "m.joblib")
    b2 = load_bundle(tmp_path / "m.joblib")
    assert b2.books.title.tolist() == b.books.title.tolist()
    P, assigned = predict_topics(b2, ["a dragon wizard on a quest"])
    assert P.shape == (1, 4) and assigned[0, 0]


def test_load_missing_bundle_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="entrenar"):
        load_bundle(tmp_path / "no.joblib")


def test_bundle_holds_out_the_same_test_books_as_run(medium_catalog, taxonomy_file):
    # Con otro filtro de libros, author_split reparte los autores distinto y el bundle de la CLI
    # entrenaba con libros que run.py y los notebooks tratan como test.
    from book_topics.data import load_books, load_descriptions, load_shelves
    from book_topics.splits import author_split, usable_books
    from book_topics.taxonomy import label_books, load_taxonomy

    books, desc = load_books(medium_catalog), load_descriptions(medium_catalog)
    labels = label_books(load_shelves(medium_catalog), load_taxonomy(taxonomy_file))
    expected = author_split(usable_books(books, desc, labels, min_words=12), seed=42)["test"]
    b = train_bundle(medium_catalog, taxonomy_file, min_df=1, calibration_cv=2, min_words=12)
    assert sorted(b.test_ids) == sorted(expected)
