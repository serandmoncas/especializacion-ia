import pandas as pd
import pytest
from book_topics.artifacts import save_bundle, train_bundle
from book_topics.cli import find_book, main, normalize


@pytest.fixture
def bundle_path(medium_catalog, taxonomy_file, tmp_path):
    path = tmp_path / "m.joblib"
    save_bundle(train_bundle(medium_catalog, taxonomy_file, min_df=1, calibration_cv=2), path)
    return path


def test_normalize_and_find():
    titles = pd.Series(["Dragon Fire (Dragons, #1)", "Dragon Ice (Dragons, #2)", "Love in Paris"])
    assert normalize("  Dragon FIRE!! ") == "dragon fire"
    assert find_book("dragon fire", titles)[0] == 0
    pos, alts = find_book("zzzz", titles)
    assert pos is None and len(alts) <= 3


def test_temas_command(bundle_path, capsys):
    assert main(["temas", "a dragon wizard and her magic sword"], bundle_path=bundle_path) == 0
    assert "Fantasía" in capsys.readouterr().out.splitlines()[0]


def test_temas_low_confidence_warning(bundle_path, capsys):
    main(["temas", "zzzz qqqq"], bundle_path=bundle_path)
    assert "baja confianza" in capsys.readouterr().out


def test_libro_unknown_title(bundle_path, capsys):
    assert main(["libro", "Nonexistent Title Xyz"], bundle_path=bundle_path) == 1
    assert "No encontré" in capsys.readouterr().out


def test_libro_ambiguous_informs_choice(bundle_path, capsys):
    assert main(["libro", "fantasy tale"], bundle_path=bundle_path) == 0
    assert "También coincide con" in capsys.readouterr().out


def test_libro_without_real_topics(bundle_path, capsys):
    assert main(["libro", "Empty Book"], bundle_path=bundle_path) == 0
    assert "sin temas asignados" in capsys.readouterr().out


def test_recomendar_explains_and_excludes_read(bundle_path, capsys):
    assert main(["recomendar", "--lei", "Dragon Fire"], bundle_path=bundle_path) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line[:1].isdigit()]
    assert len(lines) == 10 and all("Fantasía" in line for line in lines[:5])
    assert not any("Dragon Fire" in line for line in lines)


def test_recomendar_null_profile_falls_back(bundle_path, capsys):
    assert (
        main(
            ["recomendar", "--lei", "Dragon Fire", "--no-me-gusto", "Dragon Fire"],
            bundle_path=bundle_path,
        )
        == 0
    )
    assert "popularidad" in capsys.readouterr().out


def test_missing_models_message(tmp_path, capsys):
    assert main(["temas", "x"], bundle_path=tmp_path / "no.joblib", data_dir=tmp_path) == 1
    assert "entrenar" in capsys.readouterr().out
