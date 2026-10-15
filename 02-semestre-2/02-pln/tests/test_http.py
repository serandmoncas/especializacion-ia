"""Caché en disco, reintentos y que la clave nunca se filtre."""

import pytest
from research_paths.http import CachedClient, HTTPError, MissingKeyError, require_env

SECRET = "sk-secreto-123"


def _client(tmp_path, server, **kw):
    sleeps = []
    client = CachedClient(tmp_path / "cache", fetch=server, sleep=sleeps.append, **kw)
    return client, sleeps


def test_second_call_comes_from_cache(tmp_path, server_cls):
    server = server_cls([(lambda p: True, {"ok": 1})])
    client, _ = _client(tmp_path, server)
    assert client.get_json("https://x.org/api", {"q": "a"}) == {"ok": 1}
    assert client.get_json("https://x.org/api", {"q": "a"}) == {"ok": 1}
    assert len(server.calls) == 1
    assert client.stats == {"network": 1, "cache": 1}


def test_retries_429_with_exponential_backoff(tmp_path, server_cls):
    server = server_cls([(lambda p: True, [429, 503, {"ok": 2}])])
    client, sleeps = _client(tmp_path, server)
    assert client.get_json("https://x.org/api", {}) == {"ok": 2}
    assert sleeps == [1.0, 2.0]


def test_client_error_fails_immediately_and_is_not_cached(tmp_path, server_cls):
    server = server_cls([(lambda p: True, [404, {"ok": 3}])])
    client, _ = _client(tmp_path, server)
    with pytest.raises(HTTPError) as err:
        client.get_json("https://x.org/api", {"api_key": SECRET})
    assert err.value.status == 404
    assert SECRET not in str(err.value)
    assert client.get_json("https://x.org/api", {"api_key": SECRET}) == {"ok": 3}


def test_gives_up_after_max_retries(tmp_path, server_cls):
    server = server_cls([(lambda p: True, 503)])
    client, sleeps = _client(tmp_path, server, max_retries=3)
    with pytest.raises(HTTPError, match="503"):
        client.get_json("https://x.org/api", {})
    assert len(server.calls) == 3 and len(sleeps) == 2


def test_api_key_is_sent_but_never_stored(tmp_path, server_cls):
    server = server_cls([(lambda p: p.get("api_key") == SECRET, {"ok": 4})])
    client, _ = _client(tmp_path, server)
    assert client.get_json("https://x.org/api", {"api_key": SECRET, "q": "b"}) == {"ok": 4}
    files = list((tmp_path / "cache").rglob("*"))
    assert files
    for f in files:
        assert SECRET not in f.name
        if f.is_file():
            assert SECRET not in f.read_text()


def test_require_env_explains_where_to_get_the_key(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    with pytest.raises(MissingKeyError, match=r"OPENALEX_API_KEY.*openalex.org"):
        require_env("OPENALEX_API_KEY", "https://openalex.org/settings/api")
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    assert require_env("OPENALEX_API_KEY", "x") == "k"
