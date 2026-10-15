import numpy as np
import pandas as pd
from book_topics.data import load_books, load_ratings
from book_topics.splits import author_split, new_books_holdout, usable_books, user_split


def _many_books(n_authors=60, per_author=5):
    rows = [(a * per_author + k, f"autor{a}") for a in range(n_authors) for k in range(per_author)]
    return pd.DataFrame(rows, columns=["book_id", "main_author"])


def test_author_split_no_author_in_two_parts():
    books = _many_books()
    parts = author_split(books, seed=1)
    owner = books.set_index("book_id")["main_author"]
    a = {k: set(owner[v]) for k, v in parts.items()}
    assert (
        a["train"].isdisjoint(a["test"])
        and a["train"].isdisjoint(a["val"])
        and a["val"].isdisjoint(a["test"])
    )
    assert sum(len(v) for v in parts.values()) == len(books)
    assert 0.15 <= len(parts["test"]) / len(books) <= 0.25


def test_new_books_holdout_fraction_and_reproducible():
    a, b = (
        new_books_holdout(np.arange(1000), 0.1, seed=3),
        new_books_holdout(np.arange(1000), 0.1, seed=3),
    )
    assert len(a) == 100 and np.array_equal(np.sort(a), np.sort(b))


def test_user_split_uses_order_and_keeps_train(tiny_catalog):
    r = load_ratings(tiny_catalog)
    train, _, test = user_split(r, test_frac=0.25, val_frac=0.34)
    assert test[test.user_id == 1].book_id.tolist() == [3]  # orden 1, 2, 5, 3 → último
    assert set(train.user_id) == set(r.user_id)
    for u in r.user_id.unique():
        assert test[test.user_id == u].order.min() > train[train.user_id == u].order.max()


def test_author_split_tiny(tiny_catalog):
    assert sum(len(v) for v in author_split(load_books(tiny_catalog), seed=0).values()) == 6


def test_usable_books_requires_min_words_and_a_topic():
    books = pd.DataFrame({"book_id": [1, 2, 3, 4], "main_author": list("abcd")})
    desc = pd.Series({1: "one two three", 2: "one", 3: "one two three"})
    labels = pd.DataFrame({"x": [True, True, False, True]}, index=[1, 2, 3, 4])
    assert usable_books(books, desc, labels, min_words=3).book_id.tolist() == [1]
