import pandas as pd
from book_topics.data import (
    clean_description,
    load_books,
    load_descriptions,
    load_ratings,
    load_shelves,
)


def test_clean_description_strips_html_and_entities():
    raw = "A <b>dragon</b> &amp; her quest. <br />Magic   and fire."
    assert clean_description(raw) == "A dragon & her quest. Magic and fire."


def test_clean_description_handles_none_and_empty():
    assert clean_description(None) == ""
    assert clean_description("   ") == ""


def test_load_books_main_author_and_types(tiny_catalog):
    books = load_books(tiny_catalog)
    assert len(books) == 6
    assert books.loc[books.book_id == 4, "main_author"].item() == "Bo Diaz"
    assert books.book_id.dtype.kind == "i"


def test_load_ratings_order_within_user(tiny_catalog):
    r = load_ratings(tiny_catalog)
    assert r[r.user_id == 1].sort_values("order").book_id.tolist() == [1, 2, 5, 3]
    assert r.order.min() == 0


def test_load_shelves_maps_to_book_id(tiny_catalog):
    s = load_shelves(tiny_catalog)
    assert set(s.columns) == {"book_id", "shelf", "count"}
    assert s[(s.book_id == 1) & (s.shelf == "fantasy")]["count"].item() == 300


def test_load_descriptions_cleans_and_keeps_empty(tiny_catalog):
    d = load_descriptions(tiny_catalog)
    assert isinstance(d, pd.Series) and len(d) == 6
    assert d[1] == "A young dragon rider & her quest. Magic and fire."
    assert d[6] == ""
