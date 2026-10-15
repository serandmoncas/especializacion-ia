import zipfile


def test_tiny_catalog_has_all_files(tiny_catalog):
    for name in ("books.csv", "ratings.csv", "book_tags.csv", "tags.csv", "books_xml.zip"):
        assert (tiny_catalog / name).exists()
    assert len(zipfile.ZipFile(tiny_catalog / "books_xml.zip").namelist()) == 6


def test_medium_catalog_size(medium_catalog):
    assert len(zipfile.ZipFile(medium_catalog / "books_xml.zip").namelist()) == 54
    lines = (medium_catalog / "ratings.csv").read_text().splitlines()
    assert len(lines) == 1 + 40 * 15
