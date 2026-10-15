"""Descarga MovieLens 1M (GroupLens) a data/ml-1m/.

1 000 209 calificaciones (1–5 estrellas) de 6 040 usuarios sobre 3 706 películas, 2000–2003,
con fecha de cada calificación. Cada usuario tiene al menos 20 calificaciones.

Licencia (resumen, ver README del dataset): uso con fines de investigación/educación no comercial,
**no se puede redistribuir** (por eso los datos no están en el repo), citar:
F. M. Harper, J. A. Konstan, "The MovieLens Datasets: History and Context",
ACM TiiS 5(4), 2015. https://grouplens.org/datasets/movielens/1m/

Uso: uv run python 01-semestre-1/06-electiva-1/b-sistemas-de-recomendacion/data/download.py
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
WANTED = {"ratings.dat", "movies.dat", "users.dat", "README"}


def main() -> None:
    dest = DATA_DIR / "ml-1m"
    if (dest / "ratings.dat").exists():
        print(f"Ya existe: {dest}")
        return
    with urllib.request.urlopen(URL, timeout=120) as resp:
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if SHA256 is not None and digest != SHA256:
        raise RuntimeError(f"Checksum inesperado: {digest}")
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name  # solo el nombre: nada fuera de dest
            if name in WANTED:
                (dest / name).write_bytes(archive.read(member))
    print(f"MovieLens 1M → {dest} (sha256={digest})")


if __name__ == "__main__":
    main()
