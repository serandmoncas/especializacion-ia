"""Ajuste de un XLM-R diminuto (sin descargas): aprende, elige la mejor época y se guarda."""

import numpy as np
import pytest

pytest.importorskip("transformers")

from research_paths.textclf import (
    TrainConfig,
    load_classifier,
    predict_proba,
    save_classifier,
    tiny_model,
    train_classifier,
)

WORDS = ["gato", "perro", "casa", "árbol", "río", "sol", "luna", "mar"]


def _pairs(rng, n):
    a = [" ".join(rng.choice(WORDS, 4)) for _ in range(n)]
    y = rng.integers(0, 2, n)
    b = [
        ("básico elemental " if t else "avanzado complejo ") + " ".join(rng.choice(WORDS, 3))
        for t in y
    ]
    return a, b, y


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return _pairs(rng, 160), _pairs(rng, 60)


CFG = TrainConfig(
    model_name="tiny", max_length=64, epochs=8, batch_size=16, lr=3e-3, warmup_ratio=0.0
)


def test_pair_classifier_learns_and_keeps_best_epoch(data):
    (a, b, y), (va, vb, vy) = data
    model, tok = tiny_model(a + b)
    model, tok, history = train_classifier(
        a, y, va, vy, CFG, texts_b=b, val_b=vb, model=model, tokenizer=tok, log=lambda *_: None
    )
    assert len(history) == 8
    best = max(h["val_ap"] for h in history)
    assert best >= 0.95
    p = predict_proba(model, tok, va, vb, max_length=64)
    from research_paths.evaluation import average_precision

    assert average_precision(vy, p) == pytest.approx(best)


def test_single_text_mode_and_probabilities_in_range(data):
    (a, b, y), (va, vb, vy) = data
    texts = [x + " " + z for x, z in zip(a, b, strict=True)]
    val = [x + " " + z for x, z in zip(va, vb, strict=True)]
    model, tok = tiny_model(texts)
    cfg = TrainConfig(model_name="tiny", max_length=64, epochs=2, batch_size=16, lr=3e-3)
    model, tok, _ = train_classifier(
        texts, y, val, vy, cfg, model=model, tokenizer=tok, log=lambda *_: None
    )
    p = predict_proba(model, tok, val, max_length=64)
    assert p.shape == (60,) and np.all((p >= 0) & (p <= 1))


def test_save_and_load_give_identical_predictions(data, tmp_path):
    (a, b, _), (va, vb, _) = data
    model, tok = tiny_model(a + b)
    before = predict_proba(model, tok, va, vb, max_length=64)
    save_classifier(model, tok, tmp_path / "m")
    model2, tok2 = load_classifier(tmp_path / "m")
    np.testing.assert_allclose(
        predict_proba(model2, tok2, va, vb, max_length=64), before, atol=1e-6
    )
