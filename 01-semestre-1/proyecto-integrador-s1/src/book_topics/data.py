"""Carga de goodbooks-10k: libros, calificaciones, estantes y descripciones (del XML)."""

import html
import re
import zipfile
from pathlib import Path

import pandas as pd
from defusedxml.ElementTree import fromstring  # XML descargado de internet: parser seguro (sin XXE)

_TAGS = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def clean_description(raw: str | None) -> str:
    """Quita etiquetas HTML, decodifica entidades (&amp;) y normaliza espacios."""
    if not raw:
        return ""
    return _SPACES.sub(" ", html.unescape(_TAGS.sub(" ", raw))).strip()


def load_books(data_dir: str | Path) -> pd.DataFrame:
    b = pd.read_csv(Path(data_dir) / "books.csv", dtype={"isbn": str})
    authors = b["authors"].astype(str)
    return pd.DataFrame(
        {
            "book_id": b["book_id"].astype(int),
            "goodreads_book_id": b["goodreads_book_id"].astype(int),
            "title": b["title"].astype(str),
            "authors": authors,
            "main_author": authors.str.split(",").str[0].str.strip(),
            "year": b["original_publication_year"],
            "language_code": b["language_code"].fillna(""),
            "isbn": b["isbn"].fillna(""),
            "ratings_count": b["ratings_count"].astype(int),
        }
    ).sort_values("book_id", ignore_index=True)


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    r = pd.read_csv(Path(data_dir) / "ratings.csv")
    r["order"] = r.groupby("user_id").cumcount()  # el archivo viene ordenado por tiempo
    return r


def load_shelves(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    bt = pd.read_csv(data_dir / "book_tags.csv")
    tags = pd.read_csv(data_dir / "tags.csv")
    ids = load_books(data_dir)[["book_id", "goodreads_book_id"]]
    s = bt.merge(tags, on="tag_id").merge(ids, on="goodreads_book_id")
    return s.rename(columns={"tag_name": "shelf"})[["book_id", "shelf", "count"]].reset_index(
        drop=True
    )


def load_descriptions(data_dir: str | Path) -> pd.Series:
    """Descripción limpia por book_id. Cada XML se llama <goodreads_book_id>.xml."""
    data_dir = Path(data_dir)
    gid_to_book = load_books(data_dir).set_index("goodreads_book_id")["book_id"]
    out = {}
    with zipfile.ZipFile(data_dir / "books_xml.zip") as z:
        for name in z.namelist():
            if not name.endswith(".xml"):
                continue
            gid = int(Path(name).stem)
            if gid not in gid_to_book.index:
                continue
            book = fromstring(z.read(name)).find("book")
            raw = book.findtext("description") if book is not None else None
            out[int(gid_to_book[gid])] = clean_description(raw)
    return pd.Series(out, name="description").sort_index()
