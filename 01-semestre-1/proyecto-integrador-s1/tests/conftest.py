import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# permite `import run` (P/run.py) en el test de humo de los experimentos
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    (1, 1, 5),
    (1, 2, 5),
    (1, 5, 4),
    (1, 3, 2),
    (2, 3, 5),
    (2, 4, 4),
    (2, 1, 2),
    (2, 6, 3),
    (3, 5, 5),
    (3, 1, 4),
    (3, 2, 4),
    (3, 4, 3),
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
    return (
        f'<?xml version="1.0"?><GoodreadsResponse><book><id>{gid}</id><title>{title}</title>'
        f"<description><![CDATA[{description}]]></description>"
        f"<popular_shelves>{shelf_xml}</popular_shelves></book></GoodreadsResponse>"
    )


def write_catalog(path: Path, books, descriptions, shelves, ratings) -> Path:
    header = (
        "book_id,goodreads_book_id,best_book_id,work_id,books_count,isbn,isbn13,authors,"
        "original_publication_year,original_title,title,language_code,average_rating,"
        "ratings_count,work_ratings_count,work_text_reviews_count,ratings_1,ratings_2,"
        "ratings_3,ratings_4,ratings_5,image_url,small_image_url\n"
    )
    rows = [
        f'{b},{g},{g},{g},1,{isbn},,"{a}",{y},"{t}","{t}",{lang},4.0,{rc},{rc},10,1,1,1,1,1,,\n'
        for b, g, a, t, y, lang, isbn, rc in books
    ]
    (path / "books.csv").write_text(header + "".join(rows))
    (path / "ratings.csv").write_text(
        "user_id,book_id,rating\n" + "".join(f"{u},{b},{r}\n" for u, b, r in ratings)
    )
    names = sorted({n for s in shelves.values() for n, _ in s})
    tag_id = {n: i for i, n in enumerate(names)}
    (path / "tags.csv").write_text(
        "tag_id,tag_name\n" + "".join(f"{i},{n}\n" for n, i in tag_id.items())
    )
    (path / "book_tags.csv").write_text(
        "goodreads_book_id,tag_id,count\n"
        + "".join(f"{g},{tag_id[n]},{c}\n" for g, s in shelves.items() for n, c in s)
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for _, g, _, t, *_ in books:
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
                books.append(
                    (
                        bid,
                        gid,
                        f"Author {t_i}-{a}",
                        f"{shelf.title()} Tale {a}-{k}",
                        2000 + k,
                        "eng",
                        str(900 + bid),
                        int(rng.integers(50, 500)),
                    )
                )
                desc[gid] = " ".join(rng.choice(words, 8).tolist() + rng.choice(FILLER, 8).tolist())
                shelves[gid] = [
                    ("to-read", int(rng.integers(100, 400))),
                    (shelf, int(rng.integers(50, 200))),
                ]
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
