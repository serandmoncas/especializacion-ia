import importlib.util
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.prepare import prepare  # noqa: E402

RUN = Path(__file__).parents[1] / "proyecto" / "run.py"


def _load_run():
    spec = importlib.util.spec_from_file_location("vision_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_smoke_counts_m4(cocoa_raw_large, tmp_path):
    data = tmp_path / "data"
    prepare(cocoa_raw_large / "CocoaMoniliaDataSet", data / "prepared", max_side=160, crop_size=64)
    for part in ("train", "val", "test"):
        shutil.copy(cocoa_raw_large / f"{part}.txt", data / f"{part}.txt")
    report = _load_run().run(
        data, tmp_path / "out", smoke=True, device=torch.device("cpu"), models_dir=tmp_path / "m"
    )
    assert report["datos"]["m4_excluidas"] == 1
    assert report["datos"]["conflictos_publicada"] == ["IMG_20240227_090000000.jpg"]
    assert set(report["clasificacion_rafaga"]) == {
        "prevalencia",
        "color_textura",
        "sonda_lineal",
        "fine_tuning",
    }
    assert set(report["hipotesis"]) == {"H1", "H2", "H3"}
    for h in report["hipotesis"].values():
        assert {"diff", "ci", "p_value", "p_holm", "rechaza_h0", "apoya_hipotesis"} <= set(h)
        assert h["apoya_hipotesis"] == (h["rechaza_h0"] and h["diff"] > 0)
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert saved["datos"] == report["datos"]
    for name in ("clasificador.pt", "detector.pt", "modelos.json"):
        assert (tmp_path / "m" / name).exists()
