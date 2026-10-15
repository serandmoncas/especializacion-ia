import numpy as np
import pandas as pd
import pytest
from book_topics.data import load_shelves
from book_topics.taxonomy import (
    gini,
    label_books,
    load_taxonomy,
    shelf_concentration,
    topic_matrix,
)


def test_gini_extremes():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(9), 1.0]) == pytest.approx(0.9)


def test_topical_shelf_more_concentrated_than_organizational(tiny_catalog):
    conc = shelf_concentration(load_shelves(tiny_catalog), min_books=1).set_index("shelf")
    assert conc.loc["fantasy", "gini"] > conc.loc["to-read", "gini"]


def test_load_taxonomy_rejects_duplicate_shelf(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("a:\n  nombre: A\n  estantes: [x, y]\nb:\n  nombre: B\n  estantes: [y]\n")
    with pytest.raises(ValueError, match="y"):
        load_taxonomy(p)


def test_load_taxonomy_rejects_empty_topic(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("a:\n  nombre: A\n  estantes: []\n")
    with pytest.raises(ValueError, match="a"):
        load_taxonomy(p)


def test_label_books_respects_tau(tiny_catalog, taxonomy_file):
    labels = label_books(load_shelves(tiny_catalog), load_taxonomy(taxonomy_file), tau=0.1)
    assert labels.loc[1, "fantasia"] and not labels.loc[1, "romance"]
    assert labels.loc[4, "misterio"] and not labels.loc[4, "romance"]  # romance 10/220 < 0.1
    assert not labels.loc[6].any()  # solo estantes organizativos


def test_topic_matrix_zero_for_unlabeled():
    labels = pd.DataFrame({"a": [True, False]}, index=[1, 2])
    assert topic_matrix(labels, np.array([2, 1, 99])).tolist() == [[0.0], [1.0], [0.0]]


def test_real_taxonomy_shelves_exist():
    from book_topics import MODULE

    from common import data_dir

    d = data_dir(MODULE)
    if not (d / "book_tags.csv").exists():
        pytest.skip("datos no descargados")
    known = set(load_shelves(d).shelf)
    topics = load_taxonomy(d.parent / "temas.yaml")
    missing = [s for t in topics for s in t.estantes if s not in known]
    assert not missing, missing
    assert 30 <= len(topics) <= 45


def test_popularity_correlation_flags_organizational_shelf(tiny_catalog):
    # Un estante organizativo crece con cuánta gente estantea el libro; uno temático, no.
    conc = shelf_concentration(load_shelves(tiny_catalog), min_books=1).set_index("shelf")
    assert conc.loc["to-read", "corr_popularidad"] > conc.loc["fantasy", "corr_popularidad"]
    assert conc["corr_popularidad"].between(-1, 1).all()
