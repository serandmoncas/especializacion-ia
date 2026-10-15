"""Carga de AL-CPL, mapeo a ES y particiones por dominio sin conceptos compartidos."""

import pandas as pd
import pytest
from research_paths.alcpl import DOMAINS, folds, load_alcpl, map_concepts, split_fold


def _write(raw, domain, pairs, preqs):
    (raw / f"{domain}.pairs").write_text("".join(f"{a},{b}\n" for a, b in pairs))
    (raw / f"{domain}.preqs").write_text("".join(f"{a},{b}\n" for a, b in preqs))


@pytest.fixture
def raw(tmp_path):
    # «Media» aparece en dos dominios (como los 40 conceptos compartidos del real).
    _write(
        tmp_path,
        "data_mining",
        [("K-means", "Media"), ("Media", "K-means")],
        [("K-means", "Media")],
    )
    _write(
        tmp_path, "geometry", [("Área", "Longitud"), ("Longitud", "Área")], [("Área", "Longitud")]
    )
    _write(tmp_path, "physics", [("Fuerza", "Masa"), ("Masa", "Fuerza")], [("Fuerza", "Masa")])
    _write(
        tmp_path,
        "precalculus",
        [("Varianza", "Media"), ("Media", "Varianza")],
        [("Varianza", "Media")],
    )
    return tmp_path


def test_load_labels_pairs_from_preqs(raw):
    df = load_alcpl(raw)
    assert list(df.columns) == ["domain", "a", "b", "label"]
    assert len(df) == 8 and df.label.sum() == 4
    row = df[(df.a == "K-means") & (df.b == "Media")].iloc[0]
    assert row.label == 1 and row.domain == "data_mining"


def test_load_rejects_positives_missing_from_pairs(tmp_path):
    for d in DOMAINS:
        _write(tmp_path, d, [("A", "B")], [("A", "B")])
    (tmp_path / "physics.preqs").write_text("A,B\nC,D\n")
    with pytest.raises(ValueError, match="physics"):
        load_alcpl(tmp_path)


def test_map_concepts_drops_unmapped_and_collisions():
    df = pd.DataFrame(
        {
            "domain": ["d"] * 3,
            "a": ["Null_hypothesis", "AdaBoost", "Variance"],
            "b": ["Statistical_hypothesis_testing", "Mean", "Mean"],
            "label": [1, 1, 0],
        }
    )
    mapping = {
        "Null_hypothesis": "Contraste de hipótesis",
        "Statistical_hypothesis_testing": "Contraste de hipótesis",
        "AdaBoost": None,
        "Variance": "Varianza",
        "Mean": "Media",
    }
    out, stats = map_concepts(df, mapping)
    assert out[["a_es", "b_es"]].values.tolist() == [["Varianza", "Media"]]
    assert stats == {
        "concepts": 5,
        "concepts_unmapped": 1,
        "pairs": 3,
        "pairs_unmapped": 1,
        "pairs_collision": 1,
        "pairs_kept": 1,
        "positives_kept": 0,
    }


def test_folds_rotate_test_and_validation():
    fs = folds()
    assert [f.test for f in fs] == list(DOMAINS)
    assert [f.val for f in fs] == [*DOMAINS[1:], DOMAINS[0]]
    for f in fs:
        assert set(f.train) | {f.test, f.val} == set(DOMAINS)


def test_split_has_no_shared_concepts_and_no_inverted_pairs(raw):
    df, _ = map_concepts(
        load_alcpl(raw),
        {c: c for c in ["K-means", "Media", "Área", "Longitud", "Fuerza", "Masa", "Varianza"]},
    )
    for fold in folds():
        train, val, test = split_fold(df, fold)
        concepts = lambda x: set(x.a_es) | set(x.b_es)  # noqa: E731
        assert not concepts(train) & concepts(test)
        assert not concepts(val) & concepts(test)
        assert not concepts(train) & concepts(val)
        assert len(test) == (df.domain == fold.test).sum()  # test se conserva entero
        pairs = {(a, b) for a, b in zip(train.a_es, train.b_es, strict=True)}
        assert not any((b, a) in pairs for a, b in zip(test.a_es, test.b_es, strict=True))
