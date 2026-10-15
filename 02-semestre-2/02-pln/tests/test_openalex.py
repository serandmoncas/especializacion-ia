"""Cliente de OpenAlex: paginación por cursor, resumen invertido, ids y campos nulos."""

from research_paths.http import CachedClient
from research_paths.openalex import OpenAlex, abstract_text, short_id


def test_abstract_text_rebuilds_order_and_tolerates_none():
    inv = {"aprendizaje": [1], "El": [0], "refuerzo": [3], "por": [2]}
    assert abstract_text(inv) == "El aprendizaje por refuerzo"
    assert abstract_text(None) == ""
    assert abstract_text({}) == ""


def test_short_id():
    assert short_id("https://openalex.org/W123") == "W123"
    assert short_id("W9") == "W9"


def test_works_paginates_with_cursor_and_stops_at_max(tmp_path, server_cls):
    page1 = {"meta": {"next_cursor": "c2"}, "results": [{"id": f"W{i}"} for i in range(3)]}
    page2 = {"meta": {"next_cursor": "c3"}, "results": [{"id": f"W{i}"} for i in range(3, 6)]}
    server = server_cls(
        [(lambda p: p.get("cursor") == "*", page1), (lambda p: p.get("cursor") == "c2", page2)]
    )
    oa = OpenAlex(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), api_key="k")
    out = oa.works(filter="type:review", max_results=5, per_page=3)
    assert [w["id"] for w in out] == ["W0", "W1", "W2", "W3", "W4"]
    assert all("api_key=k" in c for c in server.calls)


def test_works_stops_when_cursor_is_exhausted(tmp_path, server_cls):
    page = {"meta": {"next_cursor": None}, "results": [{"id": "W1", "title": None}]}
    server = server_cls([(lambda p: True, page)])
    oa = OpenAlex(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), api_key=None)
    assert oa.works(filter="x", max_results=200) == [{"id": "W1", "title": None}]
    assert "api_key" not in server.calls[0]


def test_topics_reads_all_pages(tmp_path, server_cls):
    p1 = {"meta": {"next_cursor": "n"}, "results": [{"id": "T1"}] * 200}
    p2 = {"meta": {"next_cursor": None}, "results": [{"id": "T2"}]}
    server = server_cls(
        [(lambda p: p.get("cursor") == "*", p1), (lambda p: p.get("cursor") == "n", p2)]
    )
    oa = OpenAlex(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), api_key=None)
    assert len(oa.topics(filter="field.id:17")) == 201
