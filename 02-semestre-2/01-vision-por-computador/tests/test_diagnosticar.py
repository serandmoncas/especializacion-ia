import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.classifiers import build_resnet  # noqa: E402
from applied_cv.cocoa.detector import build_detector  # noqa: E402
from PIL import Image  # noqa: E402

DEMO = Path(__file__).parents[1] / "proyecto" / "diagnosticar.py"


def _demo():
    spec = importlib.util.spec_from_file_location("vision_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "modelos"
    d.mkdir()
    torch.save(build_resnet(4, pretrained=False).state_dict(), d / "clasificador.pt")
    torch.save(build_detector(False, 64, 96).state_dict(), d / "detector.pt")
    (d / "modelos.json").write_text(
        json.dumps({"umbral": 0.5, "det_min_size": 64, "det_max_size": 96})
    )
    return d


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "foto.jpg"
    Image.fromarray(np.full((96, 128, 3), (90, 160, 60), np.uint8)).save(path)
    return path


def test_missing_file(models_dir, tmp_path, capsys):
    assert _demo().main([str(tmp_path / "no.jpg")], models_dir=models_dir) == 2
    assert "No existe" in capsys.readouterr().out


def test_not_an_image(models_dir, tmp_path, capsys):
    text = tmp_path / "nota.txt"
    text.write_text("hola")
    assert _demo().main([str(text)], models_dir=models_dir) == 2
    assert "no es una imagen" in capsys.readouterr().out


def test_missing_models(photo, tmp_path, capsys):
    assert _demo().main([str(photo)], models_dir=tmp_path / "vacio") == 1
    assert "run.py" in capsys.readouterr().out


def test_no_pods_gives_no_diagnosis(models_dir, photo, capsys):
    assert _demo().main([str(photo), "--min-score", "1.01"], models_dir=models_dir) == 0
    assert "No se detectaron mazorcas" in capsys.readouterr().out


def test_diagnosis_writes_figure(models_dir, photo, tmp_path, capsys):
    out = tmp_path / "salida.png"
    code = _demo().main(
        [str(photo), "--min-score", "0", "--salida", str(out)], models_dir=models_dir
    )
    assert code == 0
    assert out.exists() or "No se detectaron mazorcas" in capsys.readouterr().out
