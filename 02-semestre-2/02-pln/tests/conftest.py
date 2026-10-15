"""Servidor HTTP falso para los tests (respuestas grabadas a mano), expuesto como fixture."""

import json
import urllib.parse

import pytest


class FakeServer:
    """`routes` es una lista de (predicado(params) -> bool, respuesta | [respuestas]).

    Una respuesta es un dict (200 con ese JSON) o un int (código de error). Si hay una lista, se
    consume en orden (sirve para simular 429 y luego 200).
    """

    def __init__(self, routes):
        self.routes = [(pred, resp if isinstance(resp, list) else [resp]) for pred, resp in routes]
        self.calls: list[str] = []

    def __call__(self, url: str) -> tuple[int, bytes]:
        self.calls.append(url)
        params = {
            k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()
        }
        for pred, responses in self.routes:
            if pred(params):
                resp = responses.pop(0) if len(responses) > 1 else responses[0]
                if isinstance(resp, int):
                    return resp, b""
                return 200, json.dumps(resp).encode()
        return 404, b""


@pytest.fixture
def server_cls():
    return FakeServer


class FakeOpenAlex:
    """OpenAlex en memoria; entiende los filtros que usa `readings` (y falla con los demás)."""

    def __init__(self, topics: dict[str, list[dict]], works: list[dict]):
        self.topic_list, self.all_works = topics, works
        self.calls: list[tuple[str, str | None]] = []

    def topics(self, *, filter):
        return list(self.topic_list.get(filter.split(":", 1)[1], []))

    def works(self, *, filter, search=None, sort=None, max_results=200, per_page=200, select=None):
        self.calls.append((filter, search))
        out = [w for w in self.all_works if _matches(w, filter)]
        if sort == "cited_by_count:desc":
            out.sort(key=lambda w: -(w.get("cited_by_count") or 0))
        return out[:max_results]


def _matches(work: dict, filt: str) -> bool:
    for cond in filt.split(","):
        key, val = cond.split(":", 1)
        if key == "primary_topic.id":
            ok = work["topic"] == val
        elif key == "type":
            ok = work["type"] in val.split("|")
        elif key == "publication_year":
            year = work["publication_year"] or 0
            if val.startswith("<"):
                ok = year < int(val[1:])
            else:
                lo, hi = map(int, val.split("-"))
                ok = lo <= year <= hi
        elif key == "referenced_works_count":
            ok = len(work.get("referenced_works") or []) > int(val[1:])
        elif key == "has_abstract":
            ok = bool(work.get("abstract_inverted_index"))
        else:
            raise ValueError(f"filtro no soportado por el falso: {key}")
        if not ok:
            return False
    return True


def make_work(wid, topic, kind, year, cites, abstract="", refs=()):
    return {
        "id": f"https://openalex.org/{wid}",
        "title": f"Trabajo {wid}",
        "type": kind,
        "publication_year": year,
        "language": "en",
        "cited_by_count": cites,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "abstract_inverted_index": {w: [i] for i, w in enumerate(abstract.split())} or None,
        "primary_topic": {"id": f"https://openalex.org/{topic}"},
        "topic": topic,
    }


INTRO_WORDS = "revisión panorama introducción tutorial conceptos básicos"
SPECIFIC_WORDS = "proponemos método experimento resultados mejora específica"


def build_oa_world() -> FakeOpenAlex:
    """2 campos × 3 temas. T3 y T6 no tienen revisión válida. En cada tema con revisión, sus 25
    referencias son los 25 trabajos más citados (así «citas» y «combinado» las encuentran)."""
    topics = {
        "17": [
            {"id": f"https://openalex.org/T{i}", "display_name": f"Tema {i}"} for i in (1, 2, 3)
        ],
        "26": [
            {"id": f"https://openalex.org/T{i}", "display_name": f"Tema {i}"} for i in (4, 5, 6)
        ],
    }
    works = []
    for t in range(1, 7):
        tid = f"T{t}"
        earlier = []
        for k in range(40):
            wid = f"W{t}{k:02d}"
            is_review = k % 4 == 0
            words = INTRO_WORDS if is_review else SPECIFIC_WORDS
            works.append(
                make_work(
                    wid,
                    tid,
                    "review" if is_review else "article",
                    2000 + k % 19,
                    cites=1000 - 10 * k,
                    abstract=f"{words} tema {t}" if k % 3 else "",
                )
            )
            earlier.append(wid)
        if t not in (3, 6):
            works.append(make_work(f"R{t}", tid, "review", 2020, 5000, refs=earlier[:25]))
    return FakeOpenAlex(topics, works)


@pytest.fixture
def oa_world():
    return build_oa_world()
