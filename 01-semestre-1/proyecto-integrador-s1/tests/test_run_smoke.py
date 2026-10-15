import json

from run import run


def test_run_writes_metrics(medium_catalog, taxonomy_file, tmp_path):
    out = tmp_path / "res"
    report = run(medium_catalog, out, taxonomy_file, tau=0.1, smoke=True)
    data = json.loads((out / "metrics.json").read_text())
    assert {"parte_a", "parte_b", "libros_nuevos", "hipotesis"} <= set(data)
    assert set(report["hipotesis"]) == {"H1", "H2", "H3"}
    for h in report["hipotesis"].values():
        assert {"diff", "ci", "p_value", "p_holm", "rechaza_h0", "apoya_hipotesis"} <= set(h)
        # apoyar exige significancia *y* la dirección prerregistrada (Δ > 0)
        assert h["apoya_hipotesis"] == (h["rechaza_h0"] and h["diff"] > 0)
