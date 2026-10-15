"""CLI `ruta` con mundo falso: ruta completa, desambiguación, modelos o claves faltantes."""

import io

import pytest

pytest.importorskip("transformers")

from research_paths.cli import main
from research_paths.retrieval import hashing_encoder
from research_paths.textclf import tiny_model


@pytest.fixture
def models():
    texts = ["aprendizaje por refuerzo", "probabilidad", "revisión introducción", "método"]
    return tiny_model(texts), tiny_model(texts)


def _run(argv, **kw):
    out = io.StringIO()
    code = main(argv, out=out, device="cpu", **kw)
    return code, out.getvalue()


def test_route_with_readings(wiki_world, oa_world, models):
    prereq, intro = models
    # El OpenAlex falso ignora `search` y filtra solo por tipo.
    code, text = _run(
        ["ruta", "Aprendizaje por refuerzo", "--conceptos", "3", "--lecturas", "2"],
        wiki=wiki_world,
        oa=oa_world,
        encoder=hashing_encoder(),
        prereq=prereq,
        intro=intro,
    )
    assert code == 0
    assert "Ruta para «Aprendizaje por refuerzo»" in text
    assert text.count("Lecturas:") == 3
    assert "citas" in text


def test_route_without_readings_needs_no_key(wiki_world, models, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    prereq, intro = models
    code, text = _run(
        ["ruta", "Aprendizaje por refuerzo", "--sin-lecturas"],
        wiki=wiki_world,
        encoder=hashing_encoder(),
        prereq=prereq,
        intro=intro,
    )
    assert code == 0 and "Lecturas:" not in text


def test_disambiguation_suggests_titles(wiki_world, models):
    prereq, intro = models
    code, text = _run(
        ["ruta", "Mercurio", "--sin-lecturas"],
        wiki=wiki_world,
        encoder=hashing_encoder(),
        prereq=prereq,
        intro=intro,
    )
    assert code == 1 and "¿Quisiste decir?" in text and "Mercurio (planeta)" in text


def test_missing_models_explain_what_to_run(wiki_world, tmp_path):
    code, text = _run(["ruta", "Aprendizaje por refuerzo"], wiki=wiki_world, models_dir=tmp_path)
    assert code == 2 and "run.py" in text


def test_missing_openalex_key_is_explained(wiki_world, models, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    prereq, intro = models
    code, text = _run(
        ["ruta", "Aprendizaje por refuerzo"],
        wiki=wiki_world,
        encoder=hashing_encoder(),
        prereq=prereq,
        intro=intro,
    )
    assert code == 2 and "OPENALEX_API_KEY" in text
