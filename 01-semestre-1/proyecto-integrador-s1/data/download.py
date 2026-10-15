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
FILES = {  # ruta remota → sha256
    "books.csv": "4d21d0e0433128657c752e3a3d0405ac0ec11675d6c70c64cb0abf4ab1acb76a",
    "ratings.csv": "1ee8a97172bd6d97147a30d9cfb3029ac240bd27ba2aa9365981c051c65113f8",
    "book_tags.csv": "8ff2f0659be9ddbe10bf47ba33b5b69024f893f017ac665236f15bc97d862950",
    "tags.csv": "87f316957564fa21c316a4f50afc87f07906f478f990eec027aaca00c536c449",
    "books_xml/books_xml.zip": "f9724dddfbd4d744e97977c4d631684b2aa238739893749ffc66b7cb8f5f3a66",
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
