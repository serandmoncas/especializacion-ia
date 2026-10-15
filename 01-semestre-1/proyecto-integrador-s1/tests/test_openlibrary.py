import json

import pandas as pd
from book_topics.openlibrary import agreement, fetch_subjects, subjects_to_topics


def test_subjects_to_topics_keyword_match():
    assert subjects_to_topics(["Fantasy fiction", "Juvenile fiction"]) >= {"fantasia", "juvenil"}
    assert subjects_to_topics(["nyt:series_books=2010"]) == set()


def test_fetch_uses_cache_and_injected_fetch(tmp_path):
    calls = []

    def fake(isbn):
        calls.append(isbn)
        return ["Romance"]

    cache = tmp_path / "ol.json"
    fetch_subjects(["1", "2"], cache, pause=0, fetch=fake)
    fetch_subjects(["1", "2"], cache, pause=0, fetch=fake)
    assert calls == ["1", "2"] and json.loads(cache.read_text())["1"] == ["Romance"]


def test_fetch_failure_is_skipped(tmp_path):
    def broken(isbn):
        raise OSError("sin red")

    assert fetch_subjects(["9"], tmp_path / "c.json", pause=0, fetch=broken) == {}


def test_agreement_perfect():
    labels = pd.DataFrame(
        {"romance": [True, False, True], "fantasia": [False, True, False]}, index=[1, 2, 3]
    )
    out = agreement(labels, {1: {"romance"}, 2: {"fantasia"}, 3: {"romance"}}).set_index("topic")
    assert out.loc["romance", "kappa"] == 1.0
