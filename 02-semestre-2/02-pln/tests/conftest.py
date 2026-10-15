"""Servidor HTTP falso para los tests (respuestas grabadas a mano), expuesto como fixture."""

import json
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

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


class FakeWiki:
    """Wikipedia en memoria: `pages[título] = {"extract", "links", "langlinks": {lang: título}}`."""

    def __init__(self, pages: dict[str, dict], disambiguation=()):
        self.pages, self.disambiguation = pages, set(disambiguation)

    def resolve(self, titles):
        return {t: t if t in self.pages and t not in self.disambiguation else None for t in titles}

    def extracts(self, titles):
        return {t: self.pages.get(t, {}).get("extract", "") for t in titles}

    def links(self, title):
        return list(self.pages.get(title, {}).get("links", []))

    def langlinks(self, titles, target):
        return {t: self.pages.get(t, {}).get("langlinks", {}).get(target) for t in titles}

    def search(self, query, limit=5):
        return [t for t in self.pages if query.lower() in t.lower()][:limit]


def build_wiki_world() -> FakeWiki:
    pages = {
        "Aprendizaje por refuerzo": {
            "extract": "El aprendizaje por refuerzo estudia agentes que aprenden por recompensa.",
            "links": ["Proceso de decisión de Markov", "Probabilidad", "Programación dinámica",
                      "Estados Unidos", "Recompensa"],
        },
        "Proceso de decisión de Markov": {
            "extract": "Un proceso de decisión de Markov modela decisiones con probabilidad.",
            "links": ["Probabilidad", "Aprendizaje por refuerzo"],
            "langlinks": {"en": "Markov decision process"},
        },
        "Probabilidad": {"extract": "La probabilidad mide la incertidumbre.", "links": []},
        "Programación dinámica": {
            "extract": "La programación dinámica resuelve problemas por subproblemas.",
            "links": ["Aprendizaje por refuerzo"],
        },
        "Estados Unidos": {"extract": "", "links": []},
        "Recompensa": {"extract": "La recompensa es la señal que guía al agente.", "links": []},
        "Mercurio": {"extract": "Mercurio puede referirse a…", "links": ["Mercurio (planeta)"]},
        "Mercurio (planeta)": {"extract": "Planeta.", "links": []},
    }  # fmt: skip
    return FakeWiki(pages, disambiguation={"Mercurio"})


@pytest.fixture
def wiki_world():
    return build_wiki_world()


def build_alcpl_world(raw: Path) -> tuple[FakeWiki, FakeWiki]:
    """4 dominios × 5 conceptos en cadena (nivel 0 → 4). Escribe `raw/<dominio>.pairs/.preqs` y
    devuelve (wiki_en, wiki_es). Un concepto no tiene página ES (se descarta)."""
    from research_paths.alcpl import DOMAINS

    raw.mkdir(parents=True, exist_ok=True)
    en_pages, es_pages = {}, {}
    for d, domain in enumerate(DOMAINS):
        names = [f"{domain}_c{k}" for k in range(5)]
        pairs, preqs = [], []
        for i in range(5):
            for j in range(5):
                if i > j:  # j es prerrequisito de i (nivel menor)
                    pairs += [(names[i], names[j]), (names[j], names[i])]
                    preqs.append((names[i], names[j]))
        (raw / f"{domain}.pairs").write_text("".join(f"{a},{b}\n" for a, b in pairs))
        (raw / f"{domain}.preqs").write_text("".join(f"{a},{b}\n" for a, b in preqs))
        for k, name in enumerate(names):
            es = f"Concepto {d}-{k}"
            missing_es = d == 0 and k == 4
            en_pages[name] = {
                "extract": f"Concept of level {k} in {domain}.",
                "langlinks": {} if missing_es else {"es": es},
            }
            if not missing_es:
                es_pages[es] = {
                    "extract": f"Concepto de nivel {k} " + "básico " * (5 - k),
                    "links": [f"Concepto {d}-{m}" for m in range(k)],
                }
    return FakeWiki(en_pages), FakeWiki(es_pages)


class StubJudge:
    """Juez LLM falso: dice «sí» cuando el texto de B parece más básico que el de A."""

    def judge(self, items):
        return [ta.count("básico") < tb.count("básico") for _, ta, _, tb in items]


@pytest.fixture
def worlds():
    """Constructores de mundos falsos (los tests no pueden importar `conftest` por nombre)."""
    return SimpleNamespace(alcpl=build_alcpl_world, oa=build_oa_world, judge=StubJudge)
