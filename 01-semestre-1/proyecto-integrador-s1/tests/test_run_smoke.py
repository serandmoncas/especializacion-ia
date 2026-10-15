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


def test_validate_labels_sample_depends_only_on_seed(medium_catalog, taxonomy_file, monkeypatch):
    # La muestra de Open Library no debe moverse cuando la Parte B consume más o menos azar.
    import run as run_module
    from book_topics.data import load_books, load_shelves
    from book_topics.taxonomy import label_books, load_taxonomy

    seen = []
    monkeypatch.setattr(run_module, "fetch_subjects", lambda isbns, cache: seen.append(isbns) or {})
    books = load_books(medium_catalog)
    books["isbn"] = "978" + books.book_id.astype(str).str.zfill(7)  # el fixture trae ISBN cortos
    labels = label_books(load_shelves(medium_catalog), load_taxonomy(taxonomy_file))
    run_module.validate_labels(books, labels, medium_catalog, seed=42, n=10)
    run_module.validate_labels(books, labels, medium_catalog, seed=42, n=10)
    assert len(seen) == 2 and seen[0] == seen[1] and len(seen[0]) == 10
