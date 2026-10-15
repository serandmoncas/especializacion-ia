"""Cliente de Wikipedia con respuestas grabadas: redirecciones, desambiguación, continuación."""

from research_paths.http import CachedClient
from research_paths.wiki import Wiki


def _wiki(server_cls, tmp_path, routes, lang="es"):
    return Wiki(CachedClient(tmp_path, fetch=server_cls(routes), sleep=lambda s: None), lang)


def test_resolve_follows_normalization_and_redirects_and_rejects_disambiguation(
    tmp_path, server_cls
):
    resp = {
        "query": {
            "normalized": [{"from": "aprendizaje_por_refuerzo", "to": "Aprendizaje por refuerzo"}],
            "redirects": [{"from": "Aprendizaje por refuerzo", "to": "Aprendizaje reforzado"}],
            "pages": [
                {"title": "Aprendizaje reforzado", "pageid": 1},
                {"title": "Mercurio", "pageid": 2, "pageprops": {"disambiguation": ""}},
                {"title": "Nada", "missing": True},
            ],
        }
    }
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("prop") == "pageprops", resp)])
    out = wiki.resolve(["aprendizaje_por_refuerzo", "Mercurio", "Nada"])
    assert out == {
        "aprendizaje_por_refuerzo": "Aprendizaje reforzado",
        "Mercurio": None,
        "Nada": None,
    }


def test_extracts_maps_back_to_input_titles_and_missing_is_empty(tmp_path, server_cls):
    resp = {
        "query": {
            "normalized": [{"from": "Gato_doméstico", "to": "Gato doméstico"}],
            "pages": [
                {"title": "Gato doméstico", "extract": "El gato es un mamífero."},
                {"title": "Nada", "missing": True},
            ],
        }
    }
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("prop") == "extracts", resp)])
    assert wiki.extracts(["Gato_doméstico", "Nada"]) == {
        "Gato_doméstico": "El gato es un mamífero.",
        "Nada": "",
    }


def test_links_follow_continuation(tmp_path, server_cls):
    first = {
        "continue": {"plcontinue": "1|0|B", "continue": "||"},
        "query": {"pages": [{"title": "T", "links": [{"ns": 0, "title": "A"}]}]},
    }
    second = {"query": {"pages": [{"title": "T", "links": [{"ns": 0, "title": "B"}]}]}}
    wiki = _wiki(
        server_cls,
        tmp_path,
        [
            (lambda p: p.get("prop") == "links" and "plcontinue" not in p, first),
            (lambda p: p.get("plcontinue") == "1|0|B", second),
        ],
    )
    assert wiki.links("T") == ["A", "B"]


def test_langlinks_returns_none_without_translation(tmp_path, server_cls):
    resp = {
        "query": {
            "pages": [
                {
                    "title": "Null hypothesis",
                    "langlinks": [{"lang": "es", "title": "Hipótesis nula"}],
                },
                {"title": "AdaBoost"},
            ],
            "normalized": [{"from": "Null_hypothesis", "to": "Null hypothesis"}],
        }
    }
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("prop") == "langlinks", resp)], lang="en")
    assert wiki.langlinks(["Null_hypothesis", "AdaBoost"], "es") == {
        "Null_hypothesis": "Hipótesis nula",
        "AdaBoost": None,
    }


def test_search_returns_titles(tmp_path, server_cls):
    resp = {
        "query": {"search": [{"title": "Mercurio (planeta)"}, {"title": "Mercurio (elemento)"}]}
    }
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("list") == "search", resp)])
    assert wiki.search("mercurio", 2) == ["Mercurio (planeta)", "Mercurio (elemento)"]


def test_batches_are_split(tmp_path, server_cls):
    server = server_cls([(lambda p: True, {"query": {"pages": []}})])
    wiki = Wiki(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), "es")
    wiki.extracts([f"T{i}" for i in range(45)])  # 20 + 20 + 5
    assert len(server.calls) == 3
