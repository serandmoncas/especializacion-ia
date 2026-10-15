"""Smoke de los experimentos completos sobre un mundo sintético (sin red, modelos diminutos)."""

import json

import pytest

pytest.importorskip("transformers")

from research_paths.experiments import Context, intro_agreement, run_all
from research_paths.retrieval import hashing_encoder
from research_paths.textclf import TrainConfig, tiny_model


@pytest.fixture
def ctx(tmp_path, worlds):
    wiki_en, wiki_es = worlds.alcpl(tmp_path / "data" / "alcpl")
    return Context(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "resultados",
        models_dir=tmp_path / "data" / "modelos",
        wiki_en=wiki_en,
        wiki_es=wiki_es,
        oa=worlds.oa(),
        encoder=hashing_encoder(),
        make_model=lambda texts: tiny_model(texts),
        train_cfg=TrainConfig(model_name="tiny", max_length=64, epochs=1, batch_size=8, lr=3e-3),
        judge=worlds.judge(),
        n_boot=50,
        per_field=2,
        intro_topics_per_field=2,
        fields=("17", "26"),
        log=lambda *_: None,
    )


def test_run_all_writes_complete_metrics_without_nan(ctx):
    metrics = run_all(ctx)
    on_disk = json.loads((ctx.out_dir / "metrics.json").read_text())
    assert on_disk == json.loads(json.dumps(metrics))
    cov = metrics["capa1"]["cobertura"]
    assert cov["concepts_unmapped"] == 1 and cov["pairs_kept"] < cov["pairs"]
    for model in ("prevalencia", "refd", "embeddings", "cross_es", "cross_en", "llm"):
        assert set(metrics["capa1"]["modelos"][model]) >= {"f1", "ap", "por_dominio"}
    assert metrics["capa2"]["n_temas"] == 4
    assert set(metrics["capa2"]["ordenes"]) == {"openalex", "citas", "bm25", "denso", "combinado"}
    for h in ("H1", "H2", "H3", "H4"):
        assert {"delta", "ci", "p", "p_holm", "se_apoya"} <= set(metrics["hipotesis"][h])
    for name in ("capa1_predicciones.csv", "capa2_temas.csv", "intro_muestra.csv",
                 "f1_por_dominio.png", "recall_por_orden.png"):  # fmt: skip
        assert (ctx.out_dir / name).is_file()
    assert (ctx.models_dir / "prereq-es").is_dir() and (ctx.models_dir / "intro").is_dir()


def test_without_llm_h2_is_not_evaluated(ctx):
    ctx.judge = None
    metrics = run_all(ctx)
    assert metrics["hipotesis"]["H2"] == {
        "descripcion": metrics["hipotesis"]["H2"]["descripcion"],
        "evaluada": False,
    }
    assert "llm" not in metrics["capa1"]["modelos"]


def test_intro_agreement():
    import pandas as pd

    sample = pd.DataFrame({"id": ["a", "b", "c", "d"], "p_intro": [0.9, 0.8, 0.2, 0.1]})
    labels = pd.DataFrame({"id": ["a", "b", "c", "d"], "sirve": [1, 0, 0, 0]})
    out = intro_agreement(sample, labels)
    assert out["n"] == 4 and out["positivos"] == 1 and out["ap"] == pytest.approx(1.0)
