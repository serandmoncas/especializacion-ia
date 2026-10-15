# Integrador S1 — Temas de libros como puente — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el paquete `book_topics` que predice los temas de un libro desde su descripción y los usa para recomendar libros con explicación, evaluado con hipótesis prerregistradas, más una CLI y un informe.

**Architecture:** Módulos pequeños en `src/book_topics/` (datos → taxonomía → particiones → evaluación → clasificador → recomendador → validación → artefactos → CLI), un script `run.py` que ejecuta los experimentos y escribe `resultados/`, y una CLI que carga un modelo entrenado localmente. Reutiliza `applied_recsys` (ALS implícito, PureSVD, `top_n`, `ranking_metrics`, cobertura, popularidad), `applied_stats.holm` y `applied_ml.metrics.expected_calibration_error`.

**Tech Stack:** Python 3.12, uv, numpy, pandas, scipy.sparse, scikit-learn, PyYAML, joblib, defusedxml, matplotlib, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-15-integrador-s1-libros-temas-design.md`

## Global Constraints

- Carpeta del proyecto: `01-semestre-1/proyecto-integrador-s1/` (en adelante **`P/`**).
- Paquete: `P/src/book_topics/`, registrado en `[tool.hatch.build.targets.wheel].packages` de `pyproject.toml`.
- Idioma: docs, comentarios, mensajes de la CLI y nombres visibles de temas en **español**; identificadores de código en **inglés**.
- Datos en `P/data/` (ignorado por git salvo `download.py`); modelos en `P/data/modelos/`; nunca subir datos ni modelos.
- Fuente: goodbooks-10k (Z. Zając), **CC BY-SA 4.0**, atribución en `P/README.md`.
- XML externo: parsear siempre con `defusedxml.ElementTree` (los datos vienen de internet).
- Modelos con joblib (pickle): solo cargar archivos generados localmente por `entrenar`; nunca un bundle descargado.
- Particiones: libro → temas **agrupada por autor principal**, 70/10/20; recomendación por lector, último 20 % a test y último 10 % del train restante a validación (archivo ordenado por tiempo); libros nuevos = 10 % retirado de todo el entrenamiento de la parte B.
- Hipótesis H1–H3 corregidas con Holm (`applied_stats.holm`); exploratoria sin corrección.
- Bootstrap por conglomerados: autores (parte A), lectores (parte B). 2000 remuestras, semilla 42.
- Antes de cada commit: `uv run ruff check --fix && uv run ruff format && uv run pytest`.
- Tests sin datos reales: catálogos sintéticos del `conftest.py`; los que necesitan datos reales hacen `pytest.skip` si faltan.
- Los notebooks del integrador **no** llevan `"smoke": true` (requieren datos reales).

## Review Focus

1. **Descripción con HTML, CDATA o vacía** (`<br />`, `&amp;`): texto limpio; vacía → libro fuera del clasificador, sin error. Test en Task 2 (`test_clean_description_*`, `test_load_descriptions_cleans_and_keeps_empty`).
2. **Título inexistente o ambiguo en la CLI** (`"tale"` coincide con muchos): mensaje con hasta 3 coincidencias; si es ambiguo, usar la mejor e informar cuál. Test en Task 10 (`test_libro_unknown_title`, `test_libro_ambiguous_informs_choice`).
3. **Perfil de temas nulo** (los títulos indicados no tienen temas, o “leí” y “no me gustó” se cancelan): sin NaN; recomendar por popularidad y avisarlo. Tests en Task 7 (`test_topic_scores_cosine_and_null_profile`) y Task 10 (`test_recomendar_null_profile_falls_back`).
4. **Texto sin vocabulario conocido** (`temas "zzzz qqqq"`, o en otro idioma): TF-IDF nulo → prevalencia + aviso de baja confianza. Tests en Task 6 (`test_out_of_vocabulary_text_is_low_confidence`) y Task 10 (`test_temas_low_confidence_warning`).
5. **Libro del catálogo sin temas tras el umbral** pedido con `libro`: mostrar temas predichos y decir “sin temas asignados”. Test en Task 10 (`test_libro_without_real_topics`).

---

## Estructura de archivos

```
P/
├── README.md                     # Task 12: informe (reemplaza la plantilla)
├── temas.yaml                    # Task 3: taxonomía curada
├── candidatos_estantes.csv       # Task 3: evidencia de la curaduría (Gini por estante)
├── run.py                        # Task 11: experimentos → resultados/
├── data/download.py              # Task 1
├── src/book_topics/
│   ├── __init__.py               # Task 1
│   ├── data.py                   # Task 2
│   ├── taxonomy.py               # Task 3
│   ├── splits.py                 # Task 4
│   ├── evaluation.py             # Task 5
│   ├── classify.py               # Task 6
│   ├── recommend.py              # Task 7
│   ├── openlibrary.py            # Task 8
│   ├── artifacts.py              # Task 9
│   ├── cli.py, __main__.py       # Task 10
├── notebooks/01-…, 02-…, 03-…    # Task 12
├── tests/conftest.py             # Task 1: catálogos sintéticos
└── resultados/                   # generado por run.py (se versiona: json + png)
```

Nota sobre la spec: agrega `openlibrary.py`, `artifacts.py` y `run.py` a la estructura de su sección 8; son las piezas que la spec describe (validación 4.4, persistencia de la CLI 7, resultados) en archivos propios.

---

### Task 1: Andamiaje, descarga y catálogos de prueba

**Files:**
- Create: `P/src/book_topics/__init__.py`, `P/data/download.py`, `P/tests/conftest.py`, `P/tests/test_fixture.py`
- Modify: `pyproject.toml` (packages; extra `base`: `"pyyaml>=6"`, `"joblib>=1.4"`, `"defusedxml>=0.7"`)

**Interfaces:**
- Produces:
  - `book_topics.MODULE = "01-semestre-1/proyecto-integrador-s1"`
  - fixture `tiny_catalog -> Path`: 6 libros, 3 lectores (formato real de goodbooks-10k).
  - fixture `medium_catalog -> Path`: los 6 libros + 48 generados (4 temas × 3 autores × 4 libros), 40 lectores, para entrenar modelos en tests.
  - fixture `taxonomy_file -> Path`: `temas.yaml` de 4 temas compatible con ambos catálogos.
  - En ambos catálogos los `book_id` son 1…n y los `user_id` 1…m, contiguos (como en goodbooks-10k).

- [ ] **Step 1: Registrar paquete y dependencias**

En `pyproject.toml`, agregar a `packages`:
```toml
    "01-semestre-1/proyecto-integrador-s1/src/book_topics",
```
y al extra `base`: `"pyyaml>=6",`, `"joblib>=1.4",`, `"defusedxml>=0.7",`. Luego: `uv lock && uv sync --extra base --extra dl`.

- [ ] **Step 2: `__init__.py`**

```python
"""Temas de libros como puente para recomendar (integrador S1)."""

MODULE = "01-semestre-1/proyecto-integrador-s1"
```

- [ ] **Step 3: `data/download.py`**

```python
"""Descarga goodbooks-10k (Z. Zając, CC BY-SA 4.0) a data/.

https://github.com/zygmuntz/goodbooks-10k — 10 000 libros de Goodreads, 5 976 479 calificaciones,
estantes de los lectores y el XML de cada libro (con su descripción).

Uso: uv run python 01-semestre-1/proyecto-integrador-s1/data/download.py
"""

import hashlib
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/"
FILES = {  # ruta remota → sha256 (se fijan en el Step 4)
    "books.csv": None,
    "ratings.csv": None,
    "book_tags.csv": None,
    "tags.csv": None,
    "books_xml/books_xml.zip": None,
}


def main() -> None:
    for remote, sha in FILES.items():
        out = DATA_DIR / Path(remote).name
        if out.exists():
            print(f"Ya existe: {out.name}")
            continue
        with urllib.request.urlopen(BASE + remote, timeout=300) as resp:
            content = resp.read()
        digest = hashlib.sha256(content).hexdigest()
        if sha is not None and digest != sha:
            raise RuntimeError(f"Checksum inesperado para {remote}: {digest}")
        out.write_bytes(content)
        print(f"{remote}: {len(content) / 1e6:.1f} MB (sha256={digest})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Descargar y fijar checksums**

Run: `uv run python 01-semestre-1/proyecto-integrador-s1/data/download.py`
Expected: 5 líneas con sha256. Copiar cada hash a `FILES` en lugar de `None`; volver a ejecutar → “Ya existe” ×5. `git status` no debe listar nada de `P/data/` salvo `download.py`.

- [ ] **Step 5: `tests/conftest.py`**

```python
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite `import run` en el test de Task 11

BOOKS = [  # book_id, goodreads_book_id, authors, title, year, lang, isbn, ratings_count
    (1, 101, "Ann Lee", "Dragon Fire (Dragons, #1)", 2001, "eng", "111", 900),
    (2, 102, "Ann Lee", "Dragon Ice (Dragons, #2)", 2003, "eng", "112", 700),
    (3, 103, "Bo Diaz", "Love in Paris", 2010, "eng", "113", 500),
    (4, 104, "Bo Diaz, Cy Ng", "Murder at Noon", 2012, "en-US", "114", 400),
    (5, 105, "Cy Ng", "Stars Beyond", 1999, "eng", "115", 300),
    (6, 106, "Cy Ng", "Empty Book", 2000, "", "116", 100),
]
DESCRIPTIONS = {
    101: "A young <b>dragon</b> rider &amp; her quest. <br />Magic and fire.",
    102: "The dragon war continues with ice magic and battles.",
    103: "Two strangers fall in love in Paris.",
    104: "A detective solves a murder at noon in the city.",
    105: "A starship crew explores distant galaxies and aliens.",
    106: "",
}
SHELVES = {
    101: [("to-read", 500), ("fantasy", 300), ("dragons", 120), ("owned", 40)],
    102: [("to-read", 400), ("fantasy", 250), ("dragons", 90)],
    103: [("to-read", 300), ("romance", 200), ("owned", 30)],
    104: [("to-read", 250), ("mystery", 150), ("crime", 60), ("romance", 10)],
    105: [("to-read", 200), ("science-fiction", 180), ("sci-fi", 70)],
    106: [("to-read", 50), ("owned", 5)],
}
RATINGS = [  # user_id, book_id, rating — en orden de tiempo
    (1, 1, 5), (1, 2, 5), (1, 5, 4), (1, 3, 2),
    (2, 3, 5), (2, 4, 4), (2, 1, 2), (2, 6, 3),
    (3, 5, 5), (3, 1, 4), (3, 2, 4), (3, 4, 3),
]
TOPIC_WORDS = {
    "fantasy": ["dragon", "magic", "sword", "kingdom", "wizard", "quest", "elf"],
    "romance": ["love", "kiss", "heart", "wedding", "passion", "date", "romantic"],
    "mystery": ["murder", "detective", "clue", "crime", "suspect", "police", "secret"],
    "science-fiction": ["space", "alien", "starship", "planet", "robot", "galaxy", "future"],
}
FILLER = ["the", "story", "of", "a", "life", "world", "night", "day", "city", "people"]
BASE_TOPIC = {1: "fantasy", 2: "fantasy", 3: "romance", 4: "mystery", 5: "science-fiction", 6: None}
TAXONOMY_YAML = (
    "fantasia: {nombre: Fantasía, estantes: [fantasy, dragons]}\n"
    "romance: {nombre: Romance, estantes: [romance]}\n"
    "misterio: {nombre: Misterio, estantes: [mystery, crime]}\n"
    "ciencia_ficcion: {nombre: Ciencia ficción, estantes: [science-fiction, sci-fi]}\n"
)


def _book_xml(gid, title, description, shelves):
    shelf_xml = "".join(f'<shelf name="{n}" count="{c}"/>' for n, c in shelves)
    return (f'<?xml version="1.0"?><GoodreadsResponse><book><id>{gid}</id><title>{title}</title>'
            f"<description><![CDATA[{description}]]></description>"
            f"<popular_shelves>{shelf_xml}</popular_shelves></book></GoodreadsResponse>")


def write_catalog(path: Path, books, descriptions, shelves, ratings) -> Path:
    header = ("book_id,goodreads_book_id,best_book_id,work_id,books_count,isbn,isbn13,authors,"
              "original_publication_year,original_title,title,language_code,average_rating,"
              "ratings_count,work_ratings_count,work_text_reviews_count,ratings_1,ratings_2,"
              "ratings_3,ratings_4,ratings_5,image_url,small_image_url\n")
    rows = [f'{b},{g},{g},{g},1,{isbn},,"{a}",{y},"{t}","{t}",{lang},4.0,{rc},{rc},10,1,1,1,1,1,,\n'
            for b, g, a, t, y, lang, isbn, rc in books]
    (path / "books.csv").write_text(header + "".join(rows))
    (path / "ratings.csv").write_text("user_id,book_id,rating\n" + "".join(f"{u},{b},{r}\n" for u, b, r in ratings))
    names = sorted({n for s in shelves.values() for n, _ in s})
    tag_id = {n: i for i, n in enumerate(names)}
    (path / "tags.csv").write_text("tag_id,tag_name\n" + "".join(f"{i},{n}\n" for n, i in tag_id.items()))
    (path / "book_tags.csv").write_text(
        "goodreads_book_id,tag_id,count\n"
        + "".join(f"{g},{tag_id[n]},{c}\n" for g, s in shelves.items() for n, c in s))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for b, g, a, t, *_ in books:
            z.writestr(f"books_xml/{g}.xml", _book_xml(g, t, descriptions[g], shelves[g]))
    (path / "books_xml.zip").write_bytes(buf.getvalue())
    return path


@pytest.fixture
def tiny_catalog(tmp_path):
    d = tmp_path / "tiny"
    d.mkdir()
    return write_catalog(d, BOOKS, DESCRIPTIONS, SHELVES, RATINGS)


@pytest.fixture
def medium_catalog(tmp_path):
    rng = np.random.default_rng(0)
    books, desc = list(BOOKS), dict(DESCRIPTIONS)
    shelves = {k: list(v) for k, v in SHELVES.items()}
    topic_of = dict(BASE_TOPIC)
    bid, gid = 7, 107
    for t_i, (shelf, words) in enumerate(TOPIC_WORDS.items()):
        for a in range(3):
            for k in range(4):
                books.append((bid, gid, f"Author {t_i}-{a}", f"{shelf.title()} Tale {a}-{k}", 2000 + k, "eng",
                              str(900 + bid), int(rng.integers(50, 500))))
                desc[gid] = " ".join(rng.choice(words, 8).tolist() + rng.choice(FILLER, 8).tolist())
                shelves[gid] = [("to-read", int(rng.integers(100, 400))), (shelf, int(rng.integers(50, 200)))]
                topic_of[bid] = shelf
                bid, gid = bid + 1, gid + 1
    ratings = []
    topics = list(TOPIC_WORDS)
    for u in range(1, 41):
        fav = set(rng.choice(topics, 2, replace=False))
        for b in rng.choice(np.arange(1, bid), 15, replace=False):
            r = int(rng.integers(4, 6)) if topic_of[int(b)] in fav else int(rng.integers(1, 4))
            ratings.append((u, int(b), r))
    d = tmp_path / "medium"
    d.mkdir()
    return write_catalog(d, books, desc, shelves, ratings)


@pytest.fixture
def taxonomy_file(tmp_path):
    p = tmp_path / "temas.yaml"
    p.write_text(TAXONOMY_YAML)
    return p
```

- [ ] **Step 6: `tests/test_fixture.py`**

```python
import zipfile


def test_tiny_catalog_has_all_files(tiny_catalog):
    for name in ("books.csv", "ratings.csv", "book_tags.csv", "tags.csv", "books_xml.zip"):
        assert (tiny_catalog / name).exists()
    assert len(zipfile.ZipFile(tiny_catalog / "books_xml.zip").namelist()) == 6


def test_medium_catalog_size(medium_catalog):
    assert len(zipfile.ZipFile(medium_catalog / "books_xml.zip").namelist()) == 54
    assert sum(1 for _ in open(medium_catalog / "ratings.csv")) == 1 + 40 * 15
```

Run: `uv run pytest 01-semestre-1/proyecto-integrador-s1/tests/test_fixture.py -v` → PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock 01-semestre-1/proyecto-integrador-s1/src 01-semestre-1/proyecto-integrador-s1/data/download.py 01-semestre-1/proyecto-integrador-s1/tests
git commit -m "Integrador S1: andamiaje, descarga de goodbooks-10k y catálogos de prueba"
```

---

### Task 2: Carga y limpieza de datos

**Files:**
- Create: `P/src/book_topics/data.py`, `P/tests/test_data.py`

**Interfaces:**
- Consumes: `tiny_catalog`.
- Produces:
  - `clean_description(raw: str | None) -> str`
  - `load_books(data_dir) -> pd.DataFrame` columnas `book_id, goodreads_book_id, title, authors, main_author, year, language_code, isbn, ratings_count`, ordenado por `book_id`.
  - `load_ratings(data_dir) -> pd.DataFrame` columnas `user_id, book_id, rating, order` (`order` = posición dentro del lector, 0 = más antigua).
  - `load_shelves(data_dir) -> pd.DataFrame` columnas `book_id, shelf, count`.
  - `load_descriptions(data_dir) -> pd.Series` indexada por `book_id`, texto limpio (`""` si no hay).

- [ ] **Step 1: Tests**

```python
import pandas as pd

from book_topics.data import clean_description, load_books, load_descriptions, load_ratings, load_shelves


def test_clean_description_strips_html_and_entities():
    raw = "A <b>dragon</b> &amp; her quest. <br />Magic   and fire."
    assert clean_description(raw) == "A dragon & her quest. Magic and fire."


def test_clean_description_handles_none_and_empty():
    assert clean_description(None) == ""
    assert clean_description("   ") == ""


def test_load_books_main_author_and_types(tiny_catalog):
    books = load_books(tiny_catalog)
    assert len(books) == 6
    assert books.loc[books.book_id == 4, "main_author"].item() == "Bo Diaz"
    assert books.book_id.dtype.kind == "i"


def test_load_ratings_order_within_user(tiny_catalog):
    r = load_ratings(tiny_catalog)
    assert r[r.user_id == 1].sort_values("order").book_id.tolist() == [1, 2, 5, 3]
    assert r.order.min() == 0


def test_load_shelves_maps_to_book_id(tiny_catalog):
    s = load_shelves(tiny_catalog)
    assert set(s.columns) == {"book_id", "shelf", "count"}
    assert s[(s.book_id == 1) & (s.shelf == "fantasy")]["count"].item() == 300


def test_load_descriptions_cleans_and_keeps_empty(tiny_catalog):
    d = load_descriptions(tiny_catalog)
    assert isinstance(d, pd.Series) and len(d) == 6
    assert d[1] == "A young dragon rider & her quest. Magic and fire."
    assert d[6] == ""
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest P/tests/test_data.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: `data.py`**

```python
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
    return pd.DataFrame({
        "book_id": b["book_id"].astype(int),
        "goodreads_book_id": b["goodreads_book_id"].astype(int),
        "title": b["title"].astype(str),
        "authors": authors,
        "main_author": authors.str.split(",").str[0].str.strip(),
        "year": b["original_publication_year"],
        "language_code": b["language_code"].fillna(""),
        "isbn": b["isbn"].fillna(""),
        "ratings_count": b["ratings_count"].astype(int),
    }).sort_values("book_id", ignore_index=True)


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
    return s.rename(columns={"tag_name": "shelf"})[["book_id", "shelf", "count"]].reset_index(drop=True)


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
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Verificar con datos reales**

```bash
uv run python -c "
from book_topics.data import *; d='01-semestre-1/proyecto-integrador-s1/data'
print(len(load_books(d)), len(load_ratings(d)), len(load_shelves(d)))
s=load_descriptions(d); print((s.str.split().str.len()>=20).sum())"
```
Expected: `10000 5976479 999912` (si estantes da menos, reportar cuántos `goodreads_book_id` no mapean) y `9623`.

- [ ] **Step 6: Commit** — `git commit -m "Integrador S1: carga de libros, calificaciones, estantes y descripciones"`

---

### Task 3: Taxonomía de temas y etiquetado

**Files:**
- Create: `P/src/book_topics/taxonomy.py`, `P/tests/test_taxonomy.py`, `P/temas.yaml`, `P/candidatos_estantes.csv`

**Interfaces:**
- Consumes: `load_shelves`, fixtures `tiny_catalog`, `taxonomy_file`.
- Produces:
  - `TAU: float` (umbral elegido en el Step 7; valor inicial `0.1`)
  - `gini(x: np.ndarray) -> float`
  - `shelf_concentration(shelves, min_books: int = 100) -> pd.DataFrame` (`shelf, n_books, gini`, por `gini` descendente)
  - `@dataclass(frozen=True) Topic(key: str, nombre: str, estantes: tuple[str, ...])`
  - `load_taxonomy(path) -> list[Topic]` (`ValueError` si un estante está en dos temas o un tema no tiene estantes)
  - `label_books(shelves, topics, tau: float = TAU) -> pd.DataFrame` (índice `book_id` de todos los libros con estantes; una columna bool por `Topic.key`)
  - `topic_matrix(labels, book_ids: np.ndarray) -> np.ndarray` float `[len(book_ids), n_temas]`, 0 para libros sin etiqueta

- [ ] **Step 1: Tests**

```python
import numpy as np
import pandas as pd
import pytest

from book_topics.data import load_shelves
from book_topics.taxonomy import Topic, gini, label_books, load_taxonomy, shelf_concentration, topic_matrix


def test_gini_extremes():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(9), 1.0]) == pytest.approx(0.9)


def test_topical_shelf_more_concentrated_than_organizational(tiny_catalog):
    conc = shelf_concentration(load_shelves(tiny_catalog), min_books=1).set_index("shelf")
    assert conc.loc["fantasy", "gini"] > conc.loc["to-read", "gini"]


def test_load_taxonomy_rejects_duplicate_shelf(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("a:\n  nombre: A\n  estantes: [x, y]\nb:\n  nombre: B\n  estantes: [y]\n")
    with pytest.raises(ValueError, match="y"):
        load_taxonomy(p)


def test_load_taxonomy_rejects_empty_topic(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("a:\n  nombre: A\n  estantes: []\n")
    with pytest.raises(ValueError, match="a"):
        load_taxonomy(p)


def test_label_books_respects_tau(tiny_catalog, taxonomy_file):
    labels = label_books(load_shelves(tiny_catalog), load_taxonomy(taxonomy_file), tau=0.1)
    assert labels.loc[1, "fantasia"] and not labels.loc[1, "romance"]
    assert labels.loc[4, "misterio"] and not labels.loc[4, "romance"]  # romance 10/220 < 0.1
    assert not labels.loc[6].any()  # solo estantes organizativos


def test_topic_matrix_zero_for_unlabeled():
    labels = pd.DataFrame({"a": [True, False]}, index=[1, 2])
    assert topic_matrix(labels, np.array([2, 1, 99])).tolist() == [[0.0], [1.0], [0.0]]


def test_real_taxonomy_shelves_exist():
    from book_topics import MODULE
    from common import data_dir

    d = data_dir(MODULE)
    if not (d / "book_tags.csv").exists():
        pytest.skip("datos no descargados")
    known = set(load_shelves(d).shelf)
    topics = load_taxonomy(d.parent / "temas.yaml")
    missing = [s for t in topics for s in t.estantes if s not in known]
    assert not missing, missing
    assert 30 <= len(topics) <= 45
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `taxonomy.py`**

```python
"""De estantes ruidosos a temas: concentración, taxonomía curada y etiquetado por libro."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TAU = 0.1  # se fija en el Step 7 de la Task 3 con la salida del barrido de τ


def gini(x: np.ndarray) -> float:
    """Coeficiente de Gini de valores no negativos: 0 = repartido parejo; → 1 = concentrado."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def shelf_concentration(shelves: pd.DataFrame, min_books: int = 100) -> pd.DataFrame:
    """Gini, sobre todos los libros, de la proporción que ocupa el estante entre los estantes del libro.

    Organizativo (to-read): proporción parecida en todos los libros → Gini bajo.
    Temático (fantasy): alta en unos pocos y 0 en el resto → Gini alto.
    """
    totals = shelves.groupby("book_id")["count"].sum()
    share = shelves.assign(share=shelves["count"] / shelves["book_id"].map(totals))
    n_all = shelves["book_id"].nunique()
    rows = []
    for shelf, g in share.groupby("shelf"):
        n = g["book_id"].nunique()
        if n < min_books:
            continue
        rows.append((shelf, n, gini(np.r_[g["share"].to_numpy(), np.zeros(n_all - len(g))])))
    return (pd.DataFrame(rows, columns=["shelf", "n_books", "gini"])
            .sort_values("gini", ascending=False, ignore_index=True))


@dataclass(frozen=True)
class Topic:
    key: str
    nombre: str
    estantes: tuple[str, ...]


def load_taxonomy(path: str | Path) -> list[Topic]:
    raw = yaml.safe_load(Path(path).read_text())
    topics, seen = [], {}
    for key, spec in raw.items():
        shelves = tuple(spec.get("estantes") or ())
        if not shelves:
            raise ValueError(f"El tema {key!r} no tiene estantes")
        for s in shelves:
            if s in seen:
                raise ValueError(f"El estante {s!r} está en {seen[s]!r} y en {key!r}")
            seen[s] = key
        topics.append(Topic(key, spec["nombre"], shelves))
    return topics


def label_books(shelves: pd.DataFrame, topics: list[Topic], tau: float = TAU) -> pd.DataFrame:
    """Un tema se asigna si sus estantes suman ≥ τ del total de conteos *temáticos* del libro."""
    to_topic = {s: t.key for t in topics for s in t.estantes}
    topical = shelves[shelves["shelf"].isin(to_topic)].assign(topic=lambda d: d["shelf"].map(to_topic))
    by_topic = (topical.pivot_table(index="book_id", columns="topic", values="count", aggfunc="sum", fill_value=0)
                .reindex(columns=[t.key for t in topics], fill_value=0))
    share = by_topic.div(by_topic.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    all_books = pd.Index(sorted(shelves["book_id"].unique()), name="book_id")
    return (share >= tau).reindex(all_books, fill_value=False)


def topic_matrix(labels: pd.DataFrame, book_ids: np.ndarray) -> np.ndarray:
    return labels.reindex(book_ids, fill_value=False).to_numpy(dtype=float)
```

- [ ] **Step 4: Run tests de lógica → PASS** (`test_real_taxonomy_shelves_exist` se salta o falla hasta el Step 6).

- [ ] **Step 5: Generar candidatos**

```bash
uv run python -c "
from book_topics.data import load_shelves; from book_topics.taxonomy import shelf_concentration
c = shelf_concentration(load_shelves('01-semestre-1/proyecto-integrador-s1/data'))
c.to_csv('01-semestre-1/proyecto-integrador-s1/candidatos_estantes.csv', index=False)
print(len(c)); print(c.head(60).to_string()); print(c.tail(60).to_string())"
```
Expected: ~1109 filas. Anotar el Gini por debajo del cual todos son organizativos (`to-read`, `owned`, `kindle`, `read-in-*`) y por encima del cual todos son temáticos; la franja intermedia se revisa a mano.

- [ ] **Step 6: `temas.yaml`**

Usar `candidatos_estantes.csv` para escribir 35–40 temas. Lista de partida (ajustar estantes a los nombres reales; un estante en un solo tema; solo estantes temáticos):

```yaml
fantasia: {nombre: Fantasía, estantes: [fantasy, epic-fantasy, high-fantasy, urban-fantasy]}
ciencia_ficcion: {nombre: Ciencia ficción, estantes: [science-fiction, sci-fi, scifi, space-opera]}
distopia: {nombre: Distopía, estantes: [dystopian, dystopia, post-apocalyptic]}
romance: {nombre: Romance, estantes: [romance, contemporary-romance]}
romance_historico: {nombre: Romance histórico, estantes: [historical-romance, regency]}
misterio: {nombre: Misterio, estantes: [mystery, mysteries, detective]}
thriller: {nombre: Thriller, estantes: [thriller, thrillers, suspense, mystery-thriller]}
crimen: {nombre: Crimen, estantes: [crime, crime-fiction]}
terror: {nombre: Terror, estantes: [horror]}
paranormal: {nombre: Paranormal, estantes: [paranormal, vampires, supernatural]}
ficcion_historica: {nombre: Ficción histórica, estantes: [historical-fiction, historical]}
clasicos: {nombre: Clásicos, estantes: [classics, classic, classic-literature]}
literatura: {nombre: Literatura, estantes: [literary-fiction, literature]}
juvenil: {nombre: Juvenil, estantes: [young-adult, ya, teen]}
infantil: {nombre: Infantil, estantes: [childrens, children, children-s-books, kids, picture-books]}
graficas: {nombre: Novela gráfica y cómic, estantes: [graphic-novels, comics, manga]}
humor: {nombre: Humor, estantes: [humor, humour, funny, comedy]}
poesia: {nombre: Poesía, estantes: [poetry]}
teatro: {nombre: Teatro, estantes: [plays, drama, theatre]}
cuentos: {nombre: Cuentos, estantes: [short-stories]}
no_ficcion: {nombre: No ficción, estantes: [non-fiction, nonfiction]}
biografia: {nombre: Biografía y memorias, estantes: [biography, memoir, memoirs, autobiography]}
historia: {nombre: Historia, estantes: [history]}
guerra: {nombre: Guerra, estantes: [war, wwii, world-war-ii, military]}
autoayuda: {nombre: Autoayuda, estantes: [self-help, personal-development, self-improvement]}
psicologia: {nombre: Psicología, estantes: [psychology]}
filosofia: {nombre: Filosofía, estantes: [philosophy]}
religion: {nombre: Religión y espiritualidad, estantes: [religion, christian, spirituality, christianity]}
ciencia: {nombre: Ciencia, estantes: [science, popular-science]}
negocios: {nombre: Negocios y economía, estantes: [business, economics, finance]}
politica: {nombre: Política, estantes: [politics, political]}
viajes: {nombre: Viajes, estantes: [travel]}
cocina: {nombre: Cocina, estantes: [cookbooks, cooking, food]}
deportes: {nombre: Deportes, estantes: [sports]}
aventura: {nombre: Aventura, estantes: [adventure, action]}
chick_lit: {nombre: Chick lit, estantes: [chick-lit, womens-fiction]}
lgbt: {nombre: LGBT, estantes: [lgbt, gay, lgbtq, queer]}
arte: {nombre: Arte y música, estantes: [art, music]}
```

Correr `test_real_taxonomy_shelves_exist` y corregir hasta que pase.

- [ ] **Step 7: Elegir τ**

```bash
uv run python -c "
from book_topics.data import load_shelves; from book_topics.taxonomy import load_taxonomy, label_books
d='01-semestre-1/proyecto-integrador-s1/data'; s=load_shelves(d); t=load_taxonomy(d+'/../temas.yaml')
for tau in (0.05, 0.1, 0.15, 0.2):
    L=label_books(s,t,tau); n=L.sum(axis=1)
    print(tau, 'mediana', n.median(), 'sin temas', int((n==0).sum()), 'tema mín.', int(L.sum().min()))"
```
Elegir el menor τ con mediana de temas por libro entre 2 y 4 y todos los temas con ≥ 50 libros; actualizar `TAU` en `taxonomy.py` con un comentario que cite esa salida. Si ningún τ cumple, fusionar los temas con < 50 libros en uno afín y repetir.

- [ ] **Step 8: Commit** — incluye `temas.yaml` y `candidatos_estantes.csv`. `"Integrador S1: taxonomía de temas y etiquetado de libros"`

---

### Task 4: Particiones

**Files:**
- Create: `P/src/book_topics/splits.py`, `P/tests/test_splits.py`

**Interfaces:**
- Consumes: `load_books`, `load_ratings`.
- Produces:
  - `author_split(books: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]` (`train, val, test` → `book_id`)
  - `new_books_holdout(book_ids: np.ndarray, frac: float = 0.1, seed: int = 42) -> np.ndarray`
  - `user_split(ratings, test_frac=0.2, val_frac=0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Tests**

```python
import numpy as np
import pandas as pd

from book_topics.data import load_books, load_ratings
from book_topics.splits import author_split, new_books_holdout, user_split


def _many_books(n_authors=60, per_author=5):
    rows = [(a * per_author + k, f"autor{a}") for a in range(n_authors) for k in range(per_author)]
    return pd.DataFrame(rows, columns=["book_id", "main_author"])


def test_author_split_no_author_in_two_parts():
    books = _many_books()
    parts = author_split(books, seed=1)
    owner = books.set_index("book_id")["main_author"]
    a = {k: set(owner[v]) for k, v in parts.items()}
    assert a["train"].isdisjoint(a["test"]) and a["train"].isdisjoint(a["val"]) and a["val"].isdisjoint(a["test"])
    assert sum(len(v) for v in parts.values()) == len(books)
    assert 0.15 <= len(parts["test"]) / len(books) <= 0.25


def test_new_books_holdout_fraction_and_reproducible():
    a, b = new_books_holdout(np.arange(1000), 0.1, seed=3), new_books_holdout(np.arange(1000), 0.1, seed=3)
    assert len(a) == 100 and np.array_equal(np.sort(a), np.sort(b))


def test_user_split_uses_order_and_keeps_train(tiny_catalog):
    r = load_ratings(tiny_catalog)
    train, val, test = user_split(r, test_frac=0.25, val_frac=0.34)
    assert test[test.user_id == 1].book_id.tolist() == [3]  # orden 1, 2, 5, 3 → último
    assert set(train.user_id) == set(r.user_id)
    for u in r.user_id.unique():
        assert test[test.user_id == u].order.min() > train[train.user_id == u].order.max()


def test_author_split_tiny(tiny_catalog):
    assert sum(len(v) for v in author_split(load_books(tiny_catalog), seed=0).values()) == 6
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `splits.py`**

```python
"""Particiones: por autor (libro → temas), por orden del lector (recomendación) y libros nuevos."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def author_split(books: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    """70/10/20 con todos los libros de un autor del mismo lado.

    Sin agrupar, un tomo de una serie en train y otro en test inflan las métricas: el modelo
    reconoce la serie, no el contenido.
    """
    ids, groups = books["book_id"].to_numpy(), books["main_author"].to_numpy()
    rest, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(ids, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)  # 0.125 · 0.8 = 0.1
                  .split(ids[rest], groups=groups[rest]))
    return {"train": ids[rest][tr], "val": ids[rest][va], "test": ids[test]}


def new_books_holdout(book_ids: np.ndarray, frac: float = 0.1, seed: int = 42) -> np.ndarray:
    ids = np.asarray(book_ids)
    return np.random.default_rng(seed).choice(ids, size=int(round(frac * len(ids))), replace=False)


def user_split(ratings: pd.DataFrame, test_frac: float = 0.2, val_frac: float = 0.1):
    """Por lector: último `test_frac` (según `order`) a test; del resto, último `val_frac` a validación.
    Cada lector conserva al menos una calificación en train."""
    r = ratings.sort_values(["user_id", "order"], kind="stable")
    size = r.groupby("user_id")["order"].transform("size")
    pos = r["order"]
    n_test = np.minimum(np.maximum(1, np.floor(size * test_frac)), size - 1).astype(int)
    rest = size - n_test
    n_val = np.minimum(np.floor(rest * val_frac), rest - 1).astype(int)
    is_test = pos >= size - n_test
    is_val = ~is_test & (pos >= rest - n_val)
    return r[~is_test & ~is_val], r[is_val], r[is_test]
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit** — `"Integrador S1: particiones por autor, por lector y libros nuevos"`

---

### Task 5: Métricas multi-etiqueta y bootstrap por conglomerados

**Files:**
- Create: `P/src/book_topics/evaluation.py`, `P/tests/test_evaluation.py`

**Interfaces:**
- Produces:
  - `multilabel_report(Y_true, Y_pred) -> dict` (`micro_f1: float, macro_f1: float, per_topic_f1: np.ndarray`)
  - `precision_at_k(Y_true, proba, k: int = 3) -> float`
  - `cluster_bootstrap_diff(stat_a, stat_b, groups, n_boot=2000, seed=42) -> dict` (`diff: float, ci: tuple[float, float], p_value: float`); `stat_a/stat_b: Callable[[np.ndarray], float]` reciben **índices** de observaciones.
  - `per_user_diff_bootstrap(a, b, n_boot=2000, seed=42) -> dict` (mismas claves; cada lector es un conglomerado)

- [ ] **Step 1: Tests**

```python
import numpy as np
import pytest
from sklearn.metrics import f1_score

from book_topics.evaluation import cluster_bootstrap_diff, multilabel_report, per_user_diff_bootstrap, precision_at_k


def test_multilabel_report_matches_sklearn():
    rng = np.random.default_rng(0)
    yt, yp = rng.random((50, 6)) < 0.3, rng.random((50, 6)) < 0.3
    rep = multilabel_report(yt, yp)
    assert rep["micro_f1"] == pytest.approx(f1_score(yt, yp, average="micro", zero_division=0))
    assert rep["macro_f1"] == pytest.approx(f1_score(yt, yp, average="macro", zero_division=0))


def test_precision_at_k():
    yt = np.array([[1, 0, 1, 0]], bool)
    assert precision_at_k(yt, np.array([[0.9, 0.8, 0.7, 0.1]]), k=3) == pytest.approx(2 / 3)


def test_cluster_bootstrap_wider_than_naive_with_correlated_groups():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(40), 25)
    effect = rng.normal(0, 1, 40)[groups]
    a, b = effect + rng.normal(0, 0.2, len(groups)), np.zeros(len(groups))
    clustered = cluster_bootstrap_diff(lambda i: a[i].mean(), lambda i: b[i].mean(), groups)
    naive = cluster_bootstrap_diff(lambda i: a[i].mean(), lambda i: b[i].mean(), np.arange(len(a)))

    def width(r):
        return r["ci"][1] - r["ci"][0]

    assert width(clustered) > 2 * width(naive)
    assert clustered["ci"][0] < a.mean() < clustered["ci"][1]


def test_bootstrap_p_value_small_for_clear_difference():
    rng = np.random.default_rng(2)
    res = per_user_diff_bootstrap(np.full(200, 0.5), np.full(200, 0.3) + rng.normal(0, 0.01, 200))
    assert res["diff"] == pytest.approx(0.2, abs=0.01) and res["p_value"] < 0.001
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `evaluation.py`**

```python
"""Métricas multi-etiqueta y bootstrap por conglomerados para comparaciones pareadas."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score


def multilabel_report(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    return {
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "per_topic_f1": f1_score(Y_true, Y_pred, average=None, zero_division=0),
    }


def precision_at_k(Y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-proba, axis=1)[:, :k]
    return float(np.take_along_axis(np.asarray(Y_true, bool), top, axis=1).mean())


def cluster_bootstrap_diff(stat_a: Callable[[np.ndarray], float], stat_b: Callable[[np.ndarray], float],
                           groups: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """IC 95 % y p-valor bilateral de stat_a − stat_b remuestreando *conglomerados* completos.

    Las observaciones de un mismo grupo (libros de un autor, recomendaciones a un lector) están
    correlacionadas: remuestrearlas sueltas subestima la varianza y da intervalos falsamente estrechos.
    El p-valor tiene resolución 1/n_boot: el informe muestra “< 1/n_boot” cuando toca ese piso.
    """
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = inverse.max() + 1
    members = [np.flatnonzero(inverse == g) for g in range(n_groups)]
    everything = np.arange(len(inverse))
    observed = stat_a(everything) - stat_b(everything)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, n_groups, n_groups)])
        diffs[k] = stat_a(idx) - stat_b(idx)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {"diff": float(observed), "ci": (float(lo), float(hi)), "p_value": float(min(1.0, max(p, 1 / n_boot)))}


def per_user_diff_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return cluster_bootstrap_diff(lambda i: a[i].mean(), lambda i: b[i].mean(), np.arange(len(a)), n_boot, seed)
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit** — `"Integrador S1: métricas multi-etiqueta y bootstrap por conglomerados"`

---

### Task 6: Clasificador libro → temas

**Files:**
- Create: `P/src/book_topics/classify.py`, `P/tests/test_classify.py`

**Interfaces:**
- Produces:
  - `book_text(titles: pd.Series, descriptions: pd.Series) -> pd.Series` (índice `book_id`; `"<título>. <descripción>"`)
  - `PrevalenceModel().fit(Y) -> self`; `.predict_proba(n: int) -> np.ndarray`
  - `KNNTopics(k: int = 20).fit(X, Y) -> self`; `.predict_proba(X) -> np.ndarray` (X: TF-IDF dispersa con norma L2)
  - `TextTopicModel(C: float = 4.0, max_features: int = 50_000, min_df: int = 2, calibration_cv: int = 3)`: `.fit(texts: list[str], Y) -> self`, `.transform(texts)`, `.predict_proba(texts) -> np.ndarray`, `.low_confidence(texts) -> np.ndarray[bool]`, atributo `prevalence_`
  - `fit_thresholds(Y_val, P_val, grid=None) -> np.ndarray`
  - `apply_thresholds(P, thresholds) -> np.ndarray[bool]` (al menos un tema por fila)

- [ ] **Step 1: Tests**

```python
import numpy as np
import pandas as pd

from book_topics.classify import KNNTopics, PrevalenceModel, TextTopicModel, apply_thresholds, book_text, fit_thresholds

TEXTS = (["dragon magic quest fire"] * 20 + ["love romance paris kiss"] * 20
         + ["detective murder crime clue"] * 20)
Y = np.zeros((60, 3), bool)
Y[:20, 0], Y[20:40, 1], Y[40:, 2] = True, True, True


def test_book_text_joins_title_and_description():
    assert book_text(pd.Series(["T"], index=[1]), pd.Series(["Desc."], index=[1]))[1] == "T. Desc."


def test_prevalence_model():
    p = PrevalenceModel().fit(Y).predict_proba(2)
    assert p.shape == (2, 3) and np.allclose(p, 1 / 3)


def test_text_model_learns_obvious_topics():
    p = TextTopicModel(min_df=1).fit(TEXTS, Y).predict_proba(["a dragon and magic", "a murder clue"])
    assert p[0].argmax() == 0 and p[1].argmax() == 2


def test_knn_votes_by_similarity():
    model = TextTopicModel(min_df=1).fit(TEXTS, Y)
    knn = KNNTopics(k=5).fit(model.transform(TEXTS), Y)
    assert knn.predict_proba(model.transform(["love in paris"]))[0].argmax() == 1


def test_thresholds_and_at_least_one_topic():
    P = np.array([[0.2, 0.1], [0.9, 0.8]])
    pred = apply_thresholds(P, np.array([0.5, 0.5]))
    assert pred[0].tolist() == [True, False]  # ninguno supera: el más cercano a su umbral
    assert pred[1].all()
    assert fit_thresholds(np.array([[1, 0], [1, 1]], bool), P).shape == (2,)


def test_out_of_vocabulary_text_is_low_confidence():
    model = TextTopicModel(min_df=1).fit(TEXTS, Y)
    assert model.low_confidence(["asdf qwerty zzzz", "dragon magic"]).tolist() == [True, False]
    assert np.allclose(model.predict_proba(["asdf qwerty zzzz"])[0], Y.mean(axis=0))
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `classify.py`**

```python
"""Libro → temas: prevalencia, k vecinos y TF-IDF + logística calibrada, con umbrales por tema."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier


def book_text(titles: pd.Series, descriptions: pd.Series) -> pd.Series:
    return titles.astype(str) + ". " + descriptions.reindex(titles.index).fillna("").astype(str)


class PrevalenceModel:
    def fit(self, Y: np.ndarray) -> "PrevalenceModel":
        self.prevalence_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        return np.tile(self.prevalence_, (n, 1))


class KNNTopics:
    """Temas = promedio de los temas de los k libros más parecidos, ponderado por similitud coseno."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, Y: np.ndarray) -> "KNNTopics":
        self.X_, self.Y_ = X, np.asarray(Y, dtype=float)
        return self

    def predict_proba(self, X) -> np.ndarray:
        sim = (X @ self.X_.T).toarray()  # TF-IDF con norma L2: producto punto = coseno
        k = min(self.k, sim.shape[1])
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(sim, idx, axis=1)
        votes = np.einsum("nk,nkt->nt", w, self.Y_[idx])
        total = w.sum(axis=1, keepdims=True)
        return np.where(total > 0, votes / np.where(total > 0, total, 1), self.Y_.mean(axis=0))


class TextTopicModel:
    """TF-IDF (1–2 gramas, sublineal) + logística uno-contra-todos con calibración sigmoide."""

    def __init__(self, C: float = 4.0, max_features: int = 50_000, min_df: int = 2, calibration_cv: int = 3):
        self.C, self.max_features, self.min_df, self.calibration_cv = C, max_features, min_df, calibration_cv

    def fit(self, texts: list[str], Y: np.ndarray) -> "TextTopicModel":
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=self.min_df,
                                           max_features=self.max_features, stop_words="english")
        X = self.vectorizer_.fit_transform(texts)
        base = CalibratedClassifierCV(LogisticRegression(C=self.C, max_iter=2000), method="sigmoid",
                                      cv=self.calibration_cv)
        self.model_ = OneVsRestClassifier(base, n_jobs=-1).fit(X, np.asarray(Y, dtype=int))
        self.prevalence_ = np.asarray(Y, dtype=float).mean(axis=0)
        return self

    def transform(self, texts: list[str]):
        return self.vectorizer_.transform(texts)

    def low_confidence(self, texts: list[str]) -> np.ndarray:
        """True si el texto no comparte vocabulario con el entrenamiento (vector TF-IDF nulo)."""
        return np.asarray(self.transform(texts).getnnz(axis=1) == 0)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.transform(texts)
        P = self.model_.predict_proba(X)
        P[np.asarray(X.getnnz(axis=1) == 0)] = self.prevalence_  # sin evidencia: la prevalencia
        return P


def fit_thresholds(Y_val: np.ndarray, P_val: np.ndarray, grid: np.ndarray | None = None) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    return np.array([grid[int(np.argmax([f1_score(Y_val[:, t], P_val[:, t] >= g, zero_division=0)
                                         for g in grid]))] for t in range(P_val.shape[1])])


def apply_thresholds(P: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    pred = P >= thresholds
    none = ~pred.any(axis=1)
    pred[np.flatnonzero(none), np.argmax(P[none] / thresholds, axis=1)] = True
    return pred
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit** — `"Integrador S1: clasificador libro → temas con baselines y umbrales por tema"`

---

### Task 7: Recomendación por temas, híbrida y explicaciones

**Files:**
- Create: `P/src/book_topics/recommend.py`, `P/tests/test_recommend.py`

**Interfaces:**
- Produces (T: `[n_libros, n_temas]`, fila = libro por índice interno 0…n−1):
  - `center_ratings(df: pd.DataFrame, n_users: int, n_books: int) -> sparse.csr_matrix` (`df` con `u, i, rating`)
  - `user_topic_profiles(R, T) -> np.ndarray`
  - `profile_from_titles(liked: list[int], disliked: list[int], T) -> np.ndarray`
  - `topic_scores(profiles, T) -> np.ndarray` (coseno; perfil o libro nulo → 0)
  - `zscore_rows(S) -> np.ndarray`
  - `hybrid_scores(collab, topic, alpha: float) -> np.ndarray`
  - `explain(profile, topic_vec, names: list[str], top: int = 3) -> list[str]`

- [ ] **Step 1: Tests**

```python
import numpy as np
import pandas as pd
from scipy import sparse

from book_topics.recommend import (
    center_ratings, explain, hybrid_scores, profile_from_titles, topic_scores, user_topic_profiles, zscore_rows,
)

T = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0]], float)


def test_center_ratings():
    R = center_ratings(pd.DataFrame({"u": [0, 0, 1], "i": [0, 1, 2], "rating": [5, 3, 4]}), 2, 5)
    assert R[0, 0] == 1 and R[0, 1] == -1 and R[1, 2] == 0


def test_profile_weights_topics_by_centered_rating():
    p = user_topic_profiles(sparse.csr_matrix(np.array([[1.0, 0, -1.0, 0, 0]])), T)[0]
    assert p[0] > 0 and p[2] < 0


def test_topic_scores_cosine_and_null_profile():
    S = topic_scores(np.array([[1.0, 0, 0], [0, 0, 0]]), T)
    assert S[0].argmax() == 0 and np.all(S[1] == 0) and not np.isnan(S).any() and S[0, 4] == 0


def test_profile_from_titles_cancels_to_zero():
    assert np.allclose(profile_from_titles([0], [0], T), 0)


def test_hybrid_and_zscore():
    a, b = np.array([[1.0, 2, 3]]), np.array([[3.0, 2, 1]])
    assert abs(zscore_rows(a).mean()) < 1e-12
    assert np.allclose(hybrid_scores(a, b, 1.0), zscore_rows(a))
    assert np.allclose(hybrid_scores(a, b, 0.5), 0)


def test_explain_picks_largest_contributions():
    names = ["fantasía", "aventura", "romance"]
    assert explain(np.array([0.9, 0.5, 0.1]), np.ones(3), names, top=2) == ["fantasía", "aventura"]
    assert explain(np.array([0.9, 0.5, 0.1]), np.zeros(3), names) == []
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `recommend.py`**

```python
"""Temas → libros: perfil de temas del lector, puntaje por temas, híbrido y explicaciones."""

import numpy as np
import pandas as pd
from scipy import sparse


def center_ratings(df: pd.DataFrame, n_users: int, n_books: int) -> sparse.csr_matrix:
    """Calificación − media del lector: un 3 de quien da 5 a todo es una señal negativa."""
    centered = df["rating"] - df.groupby("u")["rating"].transform("mean")
    return sparse.csr_matrix((centered.to_numpy(float), (df["u"], df["i"])), shape=(n_users, n_books))


def user_topic_profiles(R: sparse.csr_matrix, T: np.ndarray) -> np.ndarray:
    return np.asarray(R @ T)


def profile_from_titles(liked: list[int], disliked: list[int], T: np.ndarray) -> np.ndarray:
    out = np.zeros(T.shape[1])
    if liked:
        out += T[liked].sum(axis=0)
    if disliked:
        out -= T[disliked].sum(axis=0)
    return out


def topic_scores(profiles: np.ndarray, T: np.ndarray) -> np.ndarray:
    pn = np.linalg.norm(profiles, axis=1, keepdims=True)
    tn = np.linalg.norm(T, axis=1, keepdims=True)
    P = np.divide(profiles, pn, out=np.zeros_like(profiles, dtype=float), where=pn > 0)
    Tn = np.divide(T, tn, out=np.zeros_like(T, dtype=float), where=tn > 0)
    return P @ Tn.T


def zscore_rows(S: np.ndarray) -> np.ndarray:
    mu, sd = S.mean(axis=1, keepdims=True), S.std(axis=1, keepdims=True)
    return np.divide(S - mu, sd, out=np.zeros_like(S, dtype=float), where=sd > 0)


def hybrid_scores(collab: np.ndarray, topic: np.ndarray, alpha: float) -> np.ndarray:
    """α·z(colaborativo) + (1 − α)·z(temas): estandarizar por lector hace comparables las escalas."""
    return alpha * zscore_rows(collab) + (1 - alpha) * zscore_rows(topic)


def explain(profile: np.ndarray, topic_vec: np.ndarray, names: list[str], top: int = 3) -> list[str]:
    contrib = profile * topic_vec
    return [names[t] for t in np.argsort(-contrib)[:top] if contrib[t] > 0]
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit** — `"Integrador S1: recomendación por temas, híbrida y explicaciones"`

---

### Task 8: Validación de etiquetas con Open Library

**Files:**
- Create: `P/src/book_topics/openlibrary.py`, `P/tests/test_openlibrary.py`

**Interfaces:**
- Produces:
  - `SUBJECT_KEYWORDS: dict[str, list[str]]` para `fantasia, ciencia_ficcion, romance, misterio, historia, biografia, autoayuda, terror, poesia, juvenil`
  - `fetch_subjects(isbns: list[str], cache_path: Path, pause: float = 1.0, fetch: Callable[[str], list[str]] | None = None) -> dict[str, list[str]]`
  - `subjects_to_topics(subjects: list[str]) -> set[str]`
  - `agreement(labels: pd.DataFrame, ol_topics: dict[int, set[str]]) -> pd.DataFrame` (`topic, kappa, n_ours, n_ol`)

- [ ] **Step 1: Tests**

```python
import json

import pandas as pd

from book_topics.openlibrary import agreement, fetch_subjects, subjects_to_topics


def test_subjects_to_topics_keyword_match():
    assert subjects_to_topics(["Fantasy fiction", "Juvenile fiction"]) >= {"fantasia", "juvenil"}
    assert subjects_to_topics(["nyt:series_books=2010"]) == set()


def test_fetch_uses_cache_and_injected_fetch(tmp_path):
    calls = []

    def fake(isbn):
        calls.append(isbn)
        return ["Romance"]

    cache = tmp_path / "ol.json"
    fetch_subjects(["1", "2"], cache, pause=0, fetch=fake)
    fetch_subjects(["1", "2"], cache, pause=0, fetch=fake)
    assert calls == ["1", "2"] and json.loads(cache.read_text())["1"] == ["Romance"]


def test_fetch_failure_is_skipped(tmp_path):
    def broken(isbn):
        raise OSError("sin red")

    assert fetch_subjects(["9"], tmp_path / "c.json", pause=0, fetch=broken) == {}


def test_agreement_perfect():
    labels = pd.DataFrame({"romance": [True, False, True], "fantasia": [False, True, False]}, index=[1, 2, 3])
    out = agreement(labels, {1: {"romance"}, 2: {"fantasia"}, 3: {"romance"}}).set_index("topic")
    assert out.loc["romance", "kappa"] == 1.0
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `openlibrary.py`**

```python
"""Validación de las etiquetas contra los *subjects* de Open Library (fuente independiente)."""

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

SUBJECT_KEYWORDS = {
    "fantasia": ["fantasy"],
    "ciencia_ficcion": ["science fiction"],
    "romance": ["romance", "love stories"],
    "misterio": ["mystery", "detective"],
    "historia": ["history"],
    "biografia": ["biography", "memoir", "autobiography"],
    "autoayuda": ["self-help", "self-actualization"],
    "terror": ["horror"],
    "poesia": ["poetry"],
    "juvenil": ["young adult", "juvenile fiction"],
}


def _fetch_one(isbn: str) -> list[str]:
    url = "https://openlibrary.org/search.json?" + urllib.parse.urlencode(
        {"q": f"isbn:{isbn}", "fields": "subject", "limit": 1})
    with urllib.request.urlopen(url, timeout=30) as resp:
        docs = json.load(resp).get("docs", [])
    return docs[0].get("subject", []) if docs else []


def fetch_subjects(isbns: list[str], cache_path: Path, pause: float = 1.0,
                   fetch: Callable[[str], list[str]] | None = None) -> dict[str, list[str]]:
    """Consulta con caché en disco y pausa entre llamadas (Open Library pide uso moderado).
    Un error de red deja ese ISBN como `None` en la caché: se excluye, no se reintenta en bucle."""
    fetch = fetch or _fetch_one
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for isbn in isbns:
        if isbn in cache:
            continue
        try:
            cache[isbn] = fetch(isbn)
        except OSError:
            cache[isbn] = None
        cache_path.write_text(json.dumps(cache))
        time.sleep(pause)
    return {i: cache[i] for i in isbns if cache.get(i) is not None}


def subjects_to_topics(subjects: list[str]) -> set[str]:
    low = [s.lower() for s in subjects]
    return {t for t, kws in SUBJECT_KEYWORDS.items() if any(k in s for s in low for k in kws)}


def agreement(labels: pd.DataFrame, ol_topics: dict[int, set[str]]) -> pd.DataFrame:
    books = [b for b in ol_topics if b in labels.index]
    rows = []
    for t in SUBJECT_KEYWORDS:
        if t not in labels.columns:
            continue
        ours = labels.loc[books, t].astype(bool).to_numpy()
        theirs = [t in ol_topics[b] for b in books]
        same = len(set(ours)) == 1 and len(set(theirs)) == 1 and ours[0] == theirs[0]
        kappa = 1.0 if same else float(cohen_kappa_score(ours, theirs))
        rows.append((t, kappa, int(ours.sum()), int(sum(theirs))))
    return pd.DataFrame(rows, columns=["topic", "kappa", "n_ours", "n_ol"])
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit** — `"Integrador S1: validación de etiquetas con Open Library"`

---

### Task 9: Entrenamiento y persistencia de modelos

**Files:**
- Create: `P/src/book_topics/artifacts.py`, `P/tests/test_artifacts.py`

**Interfaces:**
- Consumes: `load_books, load_descriptions, load_ratings, load_shelves, load_taxonomy, label_books, topic_matrix, TAU, book_text, TextTopicModel, fit_thresholds, apply_thresholds, author_split`.
- Produces:
  - `@dataclass Bundle(books: pd.DataFrame, descriptions: pd.Series, topics: list[Topic], labels: pd.DataFrame, model: TextTopicModel, thresholds: np.ndarray, T: np.ndarray, popularity: np.ndarray)` — `T` y `popularity` alineados por posición con `books`.
  - `train_bundle(data_dir: Path, taxonomy_path: Path, *, min_df: int = 2, calibration_cv: int = 3, tau: float = TAU) -> Bundle`
  - `save_bundle(bundle: Bundle, path: Path) -> None`
  - `load_bundle(path: Path) -> Bundle` (`FileNotFoundError` con “entrenar” en el mensaje)
  - `predict_topics(bundle: Bundle, texts: list[str]) -> tuple[np.ndarray, np.ndarray]` (probabilidades, asignados)

- [ ] **Step 1: Tests**

```python
import pytest

from book_topics.artifacts import load_bundle, predict_topics, save_bundle, train_bundle


def test_train_save_load_roundtrip(medium_catalog, taxonomy_file, tmp_path):
    b = train_bundle(medium_catalog, taxonomy_file, min_df=1, calibration_cv=2)
    assert b.T.shape == (54, 4) and len(b.thresholds) == 4 and len(b.popularity) == 54
    save_bundle(b, tmp_path / "m.joblib")
    b2 = load_bundle(tmp_path / "m.joblib")
    assert b2.books.title.tolist() == b.books.title.tolist()
    P, assigned = predict_topics(b2, ["a dragon wizard on a quest"])
    assert P.shape == (1, 4) and assigned[0, 0]


def test_load_missing_bundle_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="entrenar"):
        load_bundle(tmp_path / "no.joblib")
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `artifacts.py`**

```python
"""Entrena los modelos que usa la CLI y los guarda/carga con joblib.

Seguridad: joblib usa pickle, que puede ejecutar código al cargar. Este módulo solo carga archivos
generados en esta máquina por `entrenar`; nunca abrir un bundle descargado o recibido de terceros.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from book_topics.classify import TextTopicModel, apply_thresholds, book_text, fit_thresholds
from book_topics.data import load_books, load_descriptions, load_ratings, load_shelves
from book_topics.splits import author_split
from book_topics.taxonomy import TAU, Topic, label_books, load_taxonomy, topic_matrix


@dataclass
class Bundle:
    books: pd.DataFrame
    descriptions: pd.Series
    topics: list[Topic]
    labels: pd.DataFrame
    model: TextTopicModel
    thresholds: np.ndarray
    T: np.ndarray
    popularity: np.ndarray


def train_bundle(data_dir: Path, taxonomy_path: Path, *, min_df: int = 2, calibration_cv: int = 3,
                 tau: float = TAU) -> Bundle:
    """Modelo de producción: entrena con train+val de la partición por autor; umbrales con val."""
    books = load_books(data_dir)
    desc = load_descriptions(data_dir)
    topics = load_taxonomy(taxonomy_path)
    labels = label_books(load_shelves(data_dir), topics, tau)
    has_text = desc.reindex(books.book_id).fillna("").str.len().to_numpy() > 0
    labeled = labels.reindex(books.book_id, fill_value=False).any(axis=1).to_numpy()
    parts = author_split(books[has_text & labeled], seed=42)
    texts = book_text(books.set_index("book_id")["title"], desc)
    val_ids = parts["val"] if len(parts["val"]) else parts["train"]
    fit_ids = np.r_[parts["train"], parts["val"]]
    model = TextTopicModel(min_df=min_df, calibration_cv=calibration_cv).fit(
        texts[fit_ids].tolist(), topic_matrix(labels, fit_ids).astype(bool))
    thresholds = fit_thresholds(topic_matrix(labels, val_ids).astype(bool), model.predict_proba(texts[val_ids].tolist()))
    pop = load_ratings(data_dir).groupby("book_id").size().reindex(books.book_id, fill_value=0).to_numpy(float)
    return Bundle(books, desc, topics, labels, model, thresholds, topic_matrix(labels, books.book_id.to_numpy()), pop)


def save_bundle(bundle: Bundle, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: Path) -> Bundle:
    if not Path(path).exists():
        raise FileNotFoundError(f"No hay modelos en {path}. Ejecuta primero: python -m book_topics entrenar")
    return joblib.load(path)  # solo archivos generados localmente por `entrenar` (ver docstring)


def predict_topics(bundle: Bundle, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    P = bundle.model.predict_proba(texts)
    return P, apply_thresholds(P, bundle.thresholds)
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit** — `"Integrador S1: entrenamiento y persistencia de modelos"`

---

### Task 10: CLI

**Files:**
- Create: `P/src/book_topics/cli.py`, `P/src/book_topics/__main__.py`, `P/tests/test_cli.py`

**Interfaces:**
- Consumes: `Bundle, train_bundle, save_bundle, load_bundle, predict_topics, profile_from_titles, topic_scores, explain, MODULE`.
- Produces:
  - `normalize(title: str) -> str`
  - `find_book(query: str, titles: pd.Series) -> tuple[int | None, list[str]]` (posición o None; hasta 3 alternativas)
  - `main(argv: list[str] | None = None, *, bundle_path: Path | None = None, data_dir: Path | None = None) -> int`

- [ ] **Step 1: Tests**

```python
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
    assert main(["recomendar", "--lei", "Dragon Fire", "--no-me-gusto", "Dragon Fire"], bundle_path=bundle_path) == 0
    assert "popularidad" in capsys.readouterr().out


def test_missing_models_message(tmp_path, capsys):
    assert main(["temas", "x"], bundle_path=tmp_path / "no.joblib", data_dir=tmp_path) == 1
    assert "entrenar" in capsys.readouterr().out
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `cli.py`**

```python
"""CLI: temas de un texto, ficha de un libro y recomendaciones explicadas."""

import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from book_topics import MODULE
from book_topics.artifacts import load_bundle, predict_topics, save_bundle, train_bundle
from book_topics.recommend import explain, profile_from_titles, topic_scores

_PUNCT = re.compile(r"[^\w\s]")


def normalize(title: str) -> str:
    return " ".join(_PUNCT.sub(" ", title.lower()).split())


def find_book(query: str, titles: pd.Series) -> tuple[int | None, list[str]]:
    """Exacto o prefijo → parcial → difflib. Devuelve la mejor posición y hasta 3 alternativas."""
    norm = titles.map(normalize).tolist()
    q = normalize(query)
    exact = [i for i, t in enumerate(norm) if t == q or t.startswith(q + " ")]
    if len(exact) == 1:
        return exact[0], []
    contains = exact or [i for i, t in enumerate(norm) if q and q in t]
    if contains:
        return contains[0], [titles.iloc[i] for i in contains[1:4]]
    close = difflib.get_close_matches(q, norm, n=4, cutoff=0.6)
    if close:
        return norm.index(close[0]), [titles.iloc[norm.index(c)] for c in close[1:]]
    return None, [titles.iloc[norm.index(a)] for a in difflib.get_close_matches(q, norm, n=3, cutoff=0.0)]


def _paths(bundle_path: Path | None, data_dir: Path | None) -> tuple[Path, Path]:
    if bundle_path is not None and data_dir is not None:
        return bundle_path, data_dir
    from common import data_dir as module_data_dir

    d = data_dir or module_data_dir(MODULE)
    return bundle_path or d / "modelos" / "bundle.joblib", d


def main(argv: list[str] | None = None, *, bundle_path: Path | None = None, data_dir: Path | None = None) -> int:
    parser = argparse.ArgumentParser(prog="book_topics", description="Temas de libros y recomendaciones")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("entrenar", help="entrena y guarda los modelos")
    sub.add_parser("temas", help="temas de un título o una descripción").add_argument("texto")
    sub.add_parser("libro", help="ficha de un libro del catálogo").add_argument("titulo")
    rec = sub.add_parser("recomendar", help="recomendaciones a partir de libros leídos")
    rec.add_argument("--lei", nargs="+", required=True)
    rec.add_argument("--no-me-gusto", nargs="*", default=[])
    rec.add_argument("-n", type=int, default=10)
    args = parser.parse_args(argv)
    bundle_path, data_dir = _paths(bundle_path, data_dir)

    if args.cmd == "entrenar":
        save_bundle(train_bundle(data_dir, data_dir.parent / "temas.yaml"), bundle_path)
        print(f"Modelos guardados en {bundle_path}")
        return 0
    try:
        b = load_bundle(bundle_path)
    except FileNotFoundError as e:
        print(e)
        return 1
    names = [t.nombre for t in b.topics]
    titles = b.books.title

    if args.cmd == "temas":
        text = args.texto
        pos, _ = find_book(args.texto, titles)
        if pos is not None and normalize(titles.iloc[pos]).startswith(normalize(args.texto)):
            text = f"{titles.iloc[pos]}. {b.descriptions.get(b.books.book_id.iloc[pos], '')}"
            print(f"Usando la descripción de «{titles.iloc[pos]}».")
        P, assigned = predict_topics(b, [text])
        for t in np.argsort(-P[0])[:6]:
            print(f"{'✓' if assigned[0, t] else ' '} {names[t]:<28} {P[0, t]:.2f}")
        if b.model.low_confidence([text])[0]:
            print("Aviso: el texto no comparte vocabulario con los libros conocidos (baja confianza): "
                  "se muestran las frecuencias generales de cada tema.")
        return 0

    if args.cmd == "libro":
        pos, alts = find_book(args.titulo, titles)
        if pos is None:
            print(f"No encontré «{args.titulo}». ¿Quisiste decir: {'; '.join(alts)}?")
            return 1
        print(f"{titles.iloc[pos]} — {b.books.authors.iloc[pos]}")
        if alts:
            print(f"También coincide con: {'; '.join(alts)}")
        real = [names[t] for t in np.flatnonzero(b.T[pos])]
        print("Temas reales: " + (", ".join(real) if real else "sin temas asignados (estantes insuficientes)"))
        _, assigned = predict_topics(b, [f"{titles.iloc[pos]}. {b.descriptions.get(b.books.book_id.iloc[pos], '')}"])
        print("Temas predichos: " + ", ".join(names[t] for t in np.flatnonzero(assigned[0])))
        vec = b.T[pos] if b.T[pos].any() else assigned[0].astype(float)
        scores = topic_scores(vec[None, :], b.T)[0] + 1e-9 * b.popularity
        scores[pos] = -np.inf
        print("Parecidos por temas:")
        for i, j in enumerate(np.argsort(-scores)[:10], 1):
            print(f"  {i}. {titles.iloc[j]}")
        return 0

    liked, disliked = [], []
    for queries, out in ((args.lei, liked), (args.no_me_gusto, disliked)):
        for q in queries:
            pos, alts = find_book(q, titles)
            if pos is None:
                print(f"No encontré «{q}» (se ignora). Parecidos: {'; '.join(alts)}")
                continue
            out.append(pos)
    if not liked and not disliked:
        print("Ningún título reconocido.")
        return 1
    profile = profile_from_titles(liked, disliked, b.T)
    if not np.any(profile > 0):
        print("No hay temas suficientes en lo que indicaste: recomiendo por popularidad.")
        scores = b.popularity.astype(float).copy()
    else:
        scores = topic_scores(profile[None, :], b.T)[0] + 1e-9 * b.popularity  # desempate por popularidad
    scores[list(set(liked) | set(disliked))] = -np.inf
    for i, j in enumerate(np.argsort(-scores)[: args.n], 1):
        why = explain(profile, b.T[j], names)
        print(f"{i}. {titles.iloc[j]}" + (f" — porque te gustan: {', '.join(why)}" if why else ""))
    return 0
```

`__main__.py`:
```python
import sys

from book_topics.cli import main

sys.exit(main())
```

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Probar con datos reales**

```bash
uv run python -m book_topics entrenar
uv run python -m book_topics libro "dune"
uv run python -m book_topics temas "A boy discovers he is a wizard and goes to a school of magic"
uv run python -m book_topics recomendar --lei "Dune" "Foundation" "The Hobbit"
```
Expected: Dune → ciencia ficción; la descripción del mago → fantasía/juvenil; recomendaciones con explicación y sin trazas de error.

- [ ] **Step 6: Commit** — `"Integrador S1: CLI (entrenar, temas, libro, recomendar)"`

---

### Task 11: Experimentos e hipótesis

**Files:**
- Create: `P/run.py`, `P/tests/test_run_smoke.py`

**Interfaces:**
- Consumes: todos los módulos anteriores; `applied_recsys.factorization.ImplicitALS`, `PureSVD`; `applied_recsys.metrics.top_n`, `ranking_metrics`, `catalog_coverage`, `mean_popularity_percentile`; `applied_stats.holm`; `applied_ml.metrics.expected_calibration_error`; `common.save_fig`, `set_seed`, `setup_style`.
- Produces: `run(data_dir: Path, out_dir: Path, taxonomy_path: Path, *, tau: float = TAU, max_users: int | None = None, smoke: bool = False, seed: int = 42) -> dict` que escribe `out_dir/metrics.json` y figuras, y devuelve el reporte (claves `parte_a, parte_b, libros_nuevos, hipotesis, validacion_etiquetas`).

- [ ] **Step 1: Test de humo**

```python
import json

from run import run


def test_run_writes_metrics(medium_catalog, taxonomy_file, tmp_path):
    out = tmp_path / "res"
    report = run(medium_catalog, out, taxonomy_file, tau=0.1, smoke=True)
    data = json.loads((out / "metrics.json").read_text())
    assert {"parte_a", "parte_b", "libros_nuevos", "hipotesis"} <= set(data)
    assert set(report["hipotesis"]) == {"H1", "H2", "H3"}
    for h in report["hipotesis"].values():
        assert {"diff", "ci", "p_value", "p_holm", "rechaza_h0"} <= set(h)
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: `run.py`**

```python
"""Experimentos del integrador S1: libro → temas, temas → libros y libros nuevos, con H1–H3.

Uso:
    uv run python 01-semestre-1/proyecto-integrador-s1/data/download.py
    uv run python 01-semestre-1/proyecto-integrador-s1/run.py [--max-users N]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from applied_ml.metrics import expected_calibration_error
from applied_recsys.factorization import ImplicitALS, PureSVD
from applied_recsys.metrics import catalog_coverage, mean_popularity_percentile, ranking_metrics, top_n
from applied_stats import holm
from book_topics.classify import KNNTopics, PrevalenceModel, TextTopicModel, apply_thresholds, book_text, fit_thresholds
from book_topics.data import load_books, load_descriptions, load_ratings, load_shelves
from book_topics.evaluation import cluster_bootstrap_diff, multilabel_report, per_user_diff_bootstrap, precision_at_k
from book_topics.openlibrary import agreement, fetch_subjects, subjects_to_topics
from book_topics.recommend import center_ratings, hybrid_scores, topic_scores, user_topic_profiles
from book_topics.splits import author_split, new_books_holdout, user_split
from book_topics.taxonomy import TAU, label_books, load_taxonomy, topic_matrix
from common import save_fig, set_seed, setup_style

K = 10
CHUNK = 2000


def _macro(Y, pred):
    return lambda idx: f1_score(Y[idx], pred[idx], average="macro", zero_division=0)


def _eval_ndcg(score_fn, users, exclude, relevant, n_items):
    """NDCG@k por usuario (orden de `users`), en bloques; k = min(10, candidatos − 1)."""
    k = min(K, n_items - 1)
    out = []
    for s in range(0, len(users), CHUNK):
        us = users[s : s + CHUNK]
        recs = top_n(score_fn(us), exclude[us], k)
        out.append((recs, ranking_metrics(recs, [relevant[u] for u in us], k)["ndcg"]))
    recs = np.vstack([r for r, _ in out])
    return np.concatenate([n for _, n in out]), recs


def part_a(books, desc, labels, ratings, cfg, rng):
    keys = list(labels.columns)
    has_text = desc.reindex(books.book_id).fillna("").str.split().str.len().fillna(0).to_numpy() >= cfg["min_words"]
    labeled = labels.reindex(books.book_id, fill_value=False).any(axis=1).to_numpy()
    usable = books[has_text & labeled]
    parts = author_split(usable, seed=cfg["seed"])
    texts = book_text(books.set_index("book_id")["title"], desc)
    Y = {k: topic_matrix(labels, v).astype(bool) for k, v in parts.items()}
    X = {k: texts[v].tolist() for k, v in parts.items()}

    def tuned(fit_predict_val, fit_predict_test, grid):
        best = max(grid, key=lambda g: f1_score(Y["val"], apply_thresholds(*_thr(fit_predict_val(g))),
                                                  average="macro", zero_division=0))
        P_val, P_test = fit_predict_val(best), fit_predict_test(best)
        thr = fit_thresholds(Y["val"], P_val)
        return best, P_test, apply_thresholds(P_test, thr)

    def _thr(P_val):
        return P_val, fit_thresholds(Y["val"], P_val)

    models = {}
    text_models = {}

    def logit(C):
        if C not in text_models:
            text_models[C] = TextTopicModel(C=C, min_df=cfg["min_df"], calibration_cv=cfg["cv"]).fit(X["train"], Y["train"])
        return text_models[C]

    C, P_lr, pred_lr = tuned(lambda c: logit(c).predict_proba(X["val"]), lambda c: logit(c).predict_proba(X["test"]),
                             cfg["C"])
    models["logistica"] = (P_lr, pred_lr, {"C": C})
    vec = logit(C)
    Xtr, Xva, Xte = vec.transform(X["train"]), vec.transform(X["val"]), vec.transform(X["test"])
    k, P_knn, pred_knn = tuned(lambda kk: KNNTopics(kk).fit(Xtr, Y["train"]).predict_proba(Xva),
                               lambda kk: KNNTopics(kk).fit(Xtr, Y["train"]).predict_proba(Xte), cfg["k"])
    models["knn"] = (P_knn, pred_knn, {"k": k})
    prev = PrevalenceModel().fit(Y["train"])
    P_prev = prev.predict_proba(len(parts["test"]))
    models["prevalencia"] = (P_prev, apply_thresholds(P_prev, fit_thresholds(Y["val"], prev.predict_proba(len(parts["val"])))), {})

    # Lectores → temas: factores de ALS implícito por libro (sin usar etiquetas)
    n_users, n_books = int(ratings.user_id.max()), int(books.book_id.max())
    R = sparse.csr_matrix((ratings.rating.to_numpy(float), (ratings.user_id - 1, ratings.book_id - 1)),
                          shape=(n_users, n_books))
    als = ImplicitALS(k=cfg["factors"], alpha=1.0, n_iter=cfg["als_iter"]).fit(R)
    F = {kk: als.Y[v - 1] for kk, v in parts.items()}
    P_val_r = np.column_stack([LogisticRegression(max_iter=2000).fit(F["train"], Y["train"][:, t]).predict_proba(F["val"])[:, 1]
                               if Y["train"][:, t].any() and not Y["train"][:, t].all() else np.full(len(F["val"]), Y["train"][:, t].mean())
                               for t in range(len(keys))])
    P_te_r = np.column_stack([LogisticRegression(max_iter=2000).fit(F["train"], Y["train"][:, t]).predict_proba(F["test"])[:, 1]
                              if Y["train"][:, t].any() and not Y["train"][:, t].all() else np.full(len(F["test"]), Y["train"][:, t].mean())
                              for t in range(len(keys))])
    models["lectores"] = (P_te_r, apply_thresholds(P_te_r, fit_thresholds(Y["val"], P_val_r)), {})

    groups = usable.set_index("book_id").loc[parts["test"], "main_author"].to_numpy()
    results = {}
    for name, (P, pred, params) in models.items():
        rep = multilabel_report(Y["test"], pred)
        results[name] = {"micro_f1": rep["micro_f1"], "macro_f1": rep["macro_f1"],
                         "precision_at_3": precision_at_k(Y["test"], P, 3),
                         "ece": expected_calibration_error(Y["test"].ravel(), P.ravel()),
                         "per_topic_f1": dict(zip(keys, rep["per_topic_f1"].round(4).tolist(), strict=True)),
                         "params": params}
    h1 = cluster_bootstrap_diff(_macro(Y["test"], models["logistica"][1]), _macro(Y["test"], models["knn"][1]),
                                groups, cfg["n_boot"], cfg["seed"])
    exploratory = cluster_bootstrap_diff(_macro(Y["test"], models["logistica"][1]),
                                         _macro(Y["test"], models["lectores"][1]), groups, cfg["n_boot"], cfg["seed"])
    # Análisis de errores
    prevalence = Y["train"].mean(axis=0)
    f1_lr = np.array(list(results["logistica"]["per_topic_f1"].values()))
    rho = spearmanr(prevalence, f1_lr)[0] if len(keys) > 2 else float("nan")
    fp = pred_lr & ~Y["test"]
    confusions = sorted(((int((fp[:, a] & Y["test"][:, b]).sum()), keys[a], keys[b])
                         for a in range(len(keys)) for b in range(len(keys)) if a != b), reverse=True)[:5]
    words = desc.reindex(parts["test"]).fillna("").str.split().str.len().to_numpy()
    quart = pd.qcut(words, 4, labels=False, duplicates="drop")
    by_length = {int(q): float(f1_score(Y["test"][quart == q], pred_lr[quart == q], average="micro", zero_division=0))
                 for q in np.unique(quart)}
    return {"n_libros": {k: len(v) for k, v in parts.items()}, "modelos": results,
            "spearman_prevalencia_f1": float(rho), "confusiones": confusions, "f1_por_cuartil_largo": by_length,
            "exploratoria_texto_vs_lectores": exploratory}, h1, logit(C)


def part_b_and_new(books, desc, labels, ratings, text_model_factory, cfg, rng, max_users):
    n_books = int(books.book_id.max())
    new = new_books_holdout(books.book_id.to_numpy(), 0.1, seed=cfg["seed"])
    new_idx = np.sort(new - 1)
    r = ratings.assign(u=ratings.user_id - 1, i=ratings.book_id - 1)
    old = r[~r.book_id.isin(new)]
    train, val, test = user_split(old)
    n_users = int(r.u.max()) + 1
    T = topic_matrix(labels, books.book_id.to_numpy())
    T_old = T.copy()
    T_old[new_idx] = 0
    R_tr = sparse.csr_matrix((train.rating.to_numpy(float), (train.u, train.i)), shape=(n_users, n_books))
    R_seen = sparse.csr_matrix((np.ones(len(train) + len(val)), (np.r_[train.u, val.u], np.r_[train.i, val.i])),
                               shape=(n_users, n_books))
    profiles = user_topic_profiles(center_ratings(train, n_users, n_books), T_old)
    popularity = np.asarray((R_tr != 0).sum(axis=0)).ravel().astype(float)

    def rel(df):
        pos = df[df.rating >= 4]
        return {int(u): set(g.i) for u, g in pos.groupby("u")}

    rel_val, rel_test = rel(val), rel(test)
    users_val = np.array(sorted(rel_val))
    users_test = np.array(sorted(rel_test))
    if max_users and len(users_test) > max_users:
        users_test = np.sort(rng.choice(users_test, max_users, replace=False))
        users_val = np.sort(rng.choice(users_val, min(max_users, len(users_val)), replace=False))

    def collab_candidates():
        for a in cfg["ials_alpha"]:
            yield f"ALS implícito (α={a})", ImplicitALS(k=cfg["factors"], alpha=a, n_iter=cfg["als_iter"]).fit(R_tr).scores
        for k in cfg["svd_k"]:
            yield f"PureSVD (k={k})", PureSVD(min(k, min(R_tr.shape) - 1)).fit(R_tr).scores

    best_name, best_fn, best_val = None, None, -1.0
    for name, fn in collab_candidates():
        v = _eval_ndcg(fn, users_val, R_tr, rel_val, n_books)[0].mean()
        if v > best_val:
            best_name, best_fn, best_val = name, fn, v

    def topic_fn(us):
        return topic_scores(profiles[us], T_old)

    best_alpha = max(cfg["alphas"], key=lambda a: _eval_ndcg(lambda us: hybrid_scores(best_fn(us), topic_fn(us), a),
                                                            users_val, R_tr, rel_val, n_books)[0].mean())
    scorers = {"popularidad": lambda us: np.tile(popularity, (len(us), 1)), "colaborativo": best_fn,
               "temas": topic_fn, "hibrido": lambda us: hybrid_scores(best_fn(us), topic_fn(us), best_alpha)}
    per_user, summary = {}, {}
    for name, fn in scorers.items():
        nd, recs = _eval_ndcg(fn, users_test, R_seen, rel_test, n_books)
        per_user[name] = nd
        summary[name] = {"ndcg@10": float(nd.mean()), "cobertura": catalog_coverage(recs, n_books),
                         "percentil_popularidad": mean_popularity_percentile(recs, popularity)}
    history = np.asarray((R_tr[users_test] != 0).sum(axis=1)).ravel()
    q = np.quantile(history, [0.25, 0.5, 0.75])
    bucket = np.searchsorted(q, history, side="right")
    by_history = {f"Q{b + 1}": {n: float(per_user[n][bucket == b].mean()) for n in scorers} for b in range(4)
                  if (bucket == b).any()}
    q1 = bucket == 0
    h2 = per_user_diff_bootstrap(per_user["hibrido"][q1], per_user["colaborativo"][q1], cfg["n_boot"], cfg["seed"])

    # Libros nuevos: temas predichos desde la descripción con un modelo que no los vio
    has_text = desc.reindex(books.book_id).fillna("").str.len().to_numpy() > 0
    known = books[(~books.book_id.isin(new)).to_numpy() & has_text]
    known = known[labels.reindex(known.book_id, fill_value=False).any(axis=1).to_numpy()]
    texts = book_text(books.set_index("book_id")["title"], desc)
    model = text_model_factory().fit(texts[known.book_id].tolist(), topic_matrix(labels, known.book_id.to_numpy()).astype(bool))
    new_with_text = np.array([b for b in np.sort(new) if desc.get(b, "")])
    T_new = model.predict_proba(texts[new_with_text].tolist())
    new_cols = new_with_text - 1
    rel_new = {int(u): set(g.i) for u, g in r[r.book_id.isin(new_with_text) & (r.rating >= 4)].groupby("u")}
    users_new = np.array(sorted(u for u in rel_new if profiles[u].any()))
    if max_users and len(users_new) > max_users:
        users_new = np.sort(rng.choice(users_new, max_users, replace=False))
    col_pos = {c: j for j, c in enumerate(new_cols)}
    rel_new_local = {u: {col_pos[i] for i in rel_new[u]} for u in users_new}
    pop_new = r[r.book_id.isin(new_with_text)].groupby("i").size().reindex(new_cols, fill_value=0).to_numpy(float)
    empty = sparse.csr_matrix((n_users, len(new_cols)))
    new_scorers = {"temas_predichos": lambda us: topic_scores(profiles[us], T_new),
                   "popularidad (optimista)": lambda us: np.tile(pop_new, (len(us), 1)),
                   "aleatorio": lambda us: rng.random((len(us), len(new_cols)))}
    nd_new = {n: _eval_ndcg(fn, users_new, empty, rel_new_local, len(new_cols))[0] for n, fn in new_scorers.items()}
    h3 = per_user_diff_bootstrap(nd_new["temas_predichos"], nd_new["popularidad (optimista)"], cfg["n_boot"], cfg["seed"])
    return ({"colaborativo_elegido": best_name, "alpha_hibrido": best_alpha, "n_usuarios": len(users_test),
             "modelos": summary, "por_historial": by_history},
            {"n_libros_nuevos": len(new_cols), "n_usuarios": len(users_new),
             "ndcg@10": {n: float(v.mean()) for n, v in nd_new.items()}},
            h2, h3, per_user, by_history)


def validate_labels(books, labels, data_dir, rng, n=300):
    try:
        sample = books[books.isbn.str.len() >= 9].sample(n=min(n, len(books)), random_state=int(rng.integers(1e9)))
        subjects = fetch_subjects(sample.isbn.tolist(), Path(data_dir) / "openlibrary_cache.json")
        ol = {int(b): subjects_to_topics(subjects[i]) for b, i in zip(sample.book_id, sample.isbn, strict=True)
              if i in subjects}
        return {"n_libros": len(ol), "kappa_por_tema": agreement(labels, ol).to_dict(orient="records")}
    except Exception as e:  # la validación no bloquea el resto (spec 10)
        return f"no realizada: {e}"


def run(data_dir: Path, out_dir: Path, taxonomy_path: Path, *, tau: float = TAU, max_users: int | None = None,
        smoke: bool = False, seed: int = 42) -> dict:
    rng = set_seed(seed)
    setup_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"seed": seed, "n_boot": 50 if smoke else 2000, "C": [4.0] if smoke else [1.0, 4.0, 16.0],
           "k": [5] if smoke else [10, 20, 50], "min_df": 1 if smoke else 2, "cv": 2 if smoke else 3,
           "factors": 8 if smoke else 64, "als_iter": 3 if smoke else 10, "ials_alpha": [1.0] if smoke else [0.25, 1.0],
           "svd_k": [3] if smoke else [50, 100], "alphas": [0.0, 0.5, 1.0] if smoke else [0.0, 0.25, 0.5, 0.75, 1.0],
           "min_words": 1 if smoke else 20}
    books, desc, ratings = load_books(data_dir), load_descriptions(data_dir), load_ratings(data_dir)
    labels = label_books(load_shelves(data_dir), load_taxonomy(taxonomy_path), tau)
    a, h1, _ = part_a(books, desc, labels, ratings, cfg, rng)

    def factory():
        return TextTopicModel(C=a["modelos"]["logistica"]["params"]["C"], min_df=cfg["min_df"], calibration_cv=cfg["cv"])

    b, new, h2, h3, _, by_history = part_b_and_new(books, desc, labels, ratings, factory, cfg, rng, max_users)
    p_adj = holm([h1["p_value"], h2["p_value"], h3["p_value"]])
    hyp = {}
    for (name, h), p in zip((("H1", h1), ("H2", h2), ("H3", h3)), p_adj, strict=True):
        hyp[name] = {**h, "p_holm": float(p), "rechaza_h0": bool(p < 0.05)}
    report = {"parte_a": a, "parte_b": b, "libros_nuevos": new, "hipotesis": hyp,
              "validacion_etiquetas": "omitida (smoke)" if smoke else validate_labels(books, labels, data_dir, rng)}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    keys = list(labels.columns)
    f1 = a["modelos"]["logistica"]["per_topic_f1"]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.25 * len(keys))))
    order = sorted(f1, key=f1.get)
    ax.barh(order, [f1[k] for k in order])
    ax.set(xlabel="F1 en test", title="Libro → temas: F1 por tema (logística)")
    save_fig(fig, out_dir / "f1_por_tema.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(next(iter(by_history.values())))
    for j, n in enumerate(names):
        ax.bar(np.arange(len(by_history)) + (j - 1.5) * 0.2, [by_history[q][n] for q in by_history], 0.2, label=n)
    ax.set_xticks(range(len(by_history)), list(by_history))
    ax.set(ylabel="NDCG@10", title="Recomendación por cuartil de historial")
    ax.legend(fontsize=8)
    save_fig(fig, out_dir / "ndcg_por_historial.png")
    plt.close("all")
    return report


def main() -> None:
    from book_topics import MODULE
    from common import data_dir

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()
    d = data_dir(MODULE)
    report = run(d, Path(__file__).resolve().parent / "resultados", d.parent / "temas.yaml", max_users=args.max_users)
    for name, h in report["hipotesis"].items():
        print(f"{name}: Δ = {h['diff']:+.4f} IC [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]"
              f" p = {h['p_value']:.4f} p_Holm = {h['p_holm']:.4f} → {'rechaza' if h['rechaza_h0'] else 'no rechaza'} H0")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, verify PASS** (test de humo). Si algún paso falla en el catálogo mediano (p. ej. un tema sin positivos en train de la parte A), ajustar el catálogo del `conftest` para que cada tema tenga ≥ 3 libros en train, no el código de `run.py`.

- [ ] **Step 5: Ejecutar con datos reales**

Run: `uv run python 01-semestre-1/proyecto-integrador-s1/run.py`
Expected: `resultados/metrics.json` completo y las 3 líneas de hipótesis. Si la parte B tarda > 20 min, repetir con `--max-users 20000` y anotarlo en el informe.

- [ ] **Step 6: Verificar cada afirmación** que irá al informe contra `metrics.json`.

- [ ] **Step 7: Commit** — incluye `resultados/*.json` y `resultados/*.png`. `"Integrador S1: experimentos e hipótesis"`

---

### Task 12: Notebooks e informe

**Files:**
- Create: `P/notebooks/01-taxonomia-y-etiquetas.ipynb`, `P/notebooks/02-libro-a-temas.ipynb`, `P/notebooks/03-recomendacion-y-libros-nuevos.ipynb`
- Modify: `P/README.md` (reemplaza la plantilla), `README.md` raíz (fila del integrador S1), `CLAUDE.md`

- [ ] **Step 1: Notebook 01 — celdas**

```python
# 1. Setup
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from book_topics import MODULE
from book_topics.data import load_shelves
from book_topics.taxonomy import TAU, label_books, load_taxonomy
from common import data_dir, setup_style

setup_style()
d = data_dir(MODULE)
P = d.parent
cand = pd.read_csv(P / "candidatos_estantes.csv")
topics = load_taxonomy(P / "temas.yaml")
labels = label_books(load_shelves(d), topics, TAU)
report = json.loads((P / "resultados" / "metrics.json").read_text())
```
```python
# 2. Gini de los estantes: organizativos vs. temáticos (anotar los ejemplos extremos)
fig, ax = plt.subplots()
ax.hist(cand.gini, bins=50)
for name in ("to-read", "owned", "fantasy", "romance"):
    row = cand[cand.shelf == name]
    if len(row):
        ax.axvline(row.gini.item(), ls=":", label=name)
ax.set(xlabel="Gini", ylabel="estantes")
ax.legend();
```
```python
# 3. Taxonomía final y libros por tema
counts = labels.sum().sort_values()
print(len(topics), "temas;", "mediana de temas por libro:", labels.sum(axis=1).median())
counts.plot.barh(figsize=(6, 9));
```
```python
# 4. Acuerdo con Open Library
pd.DataFrame(report["validacion_etiquetas"]["kappa_por_tema"]) if isinstance(report["validacion_etiquetas"], dict) else report["validacion_etiquetas"]
```
Markdown entre celdas con la lectura de cada resultado (escrita después de ejecutar).

- [ ] **Step 2: Notebook 02 — celdas**

```python
import json
from pathlib import Path

import pandas as pd

from book_topics import MODULE
from common import data_dir

report = json.loads((data_dir(MODULE).parent / "resultados" / "metrics.json").read_text())
a = report["parte_a"]
pd.DataFrame({m: {k: v for k, v in r.items() if k in ("micro_f1", "macro_f1", "precision_at_3", "ece")}
              for m, r in a["modelos"].items()}).T
```
```python
report["hipotesis"]["H1"], a["exploratoria_texto_vs_lectores"]
```
```python
print("Spearman prevalencia vs F1:", a["spearman_prevalencia_f1"])
pd.DataFrame(a["confusiones"], columns=["falsos positivos", "predicho", "real"])
```
```python
pd.Series(a["f1_por_cuartil_largo"], name="F1-micro por cuartil de largo de descripción")
```
Más 3 ejemplos de libros de test (acierto, error, ambiguo) usando `load_bundle` y `predict_topics`.

- [ ] **Step 3: Notebook 03 — celdas**

```python
import json

import pandas as pd

from book_topics import MODULE
from common import data_dir

report = json.loads((data_dir(MODULE).parent / "resultados" / "metrics.json").read_text())
b, new = report["parte_b"], report["libros_nuevos"]
pd.DataFrame(b["modelos"]).T, b["colaborativo_elegido"], b["alpha_hibrido"]
```
```python
pd.DataFrame(b["por_historial"]).T
```
```python
report["hipotesis"]["H2"], report["hipotesis"]["H3"], new
```
```python
import subprocess
print(subprocess.run(["python", "-m", "book_topics", "recomendar", "--lei", "Dune", "Foundation"],
                     capture_output=True, text=True).stdout)
```

- [ ] **Step 4: Ejecutar** `uv run python scripts/run_notebooks.py --inplace P/notebooks/*.ipynb` (requiere datos y `entrenar`) y escribir las celdas markdown de lectura con las cifras de la salida.

- [ ] **Step 5: `P/README.md`** — secciones: Pregunta; Datos y atribución (goodbooks-10k, Z. Zając, CC BY-SA 4.0, enlace); Diseño (taxonomía con Gini y τ, particiones por autor/lector/libros nuevos, modelos); Hipótesis (tabla: H, Δ, IC 95 %, p, p Holm, decisión — copiada de `metrics.json`); Resultados de las partes A y B y libros nuevos; Validación de etiquetas (κ); Lectura; Limitaciones (estantes como etiquetas, evaluación offline, solo inglés, `--max-users` si se usó); Uso de la CLI; Preguntas para PLN (S2): ¿cuánto mejora un transformer al TF-IDF?, ¿descripciones en español?

- [ ] **Step 6: Raíz y `CLAUDE.md`**

En `CLAUDE.md`, agregar bajo “Convenciones técnicas”:
```markdown
- **Proyectos integradores**: sus notebooks usan datos reales y no se marcan `smoke`; la
  corrección del código se cubre con tests sobre catálogos sintéticos (`tests/conftest.py`).
```
En `README.md` raíz, fila del integrador S1: `Temas de libros como puente para recomendar (goodbooks-10k) | 🚧 (falta CI en verde)`.

- [ ] **Step 7: CI local** — `uv run ruff check && uv run ruff format --check && uv run pytest && uv run python scripts/run_notebooks.py`.

- [ ] **Step 8: Commit y push**; seguir el CI; al pasar, cambiar la fila a ✅ y hacer commit.
