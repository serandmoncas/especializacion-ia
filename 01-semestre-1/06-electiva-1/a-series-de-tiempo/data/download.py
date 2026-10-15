"""Descarga UCI Bike Sharing (Capital Bikeshare, Washington D.C., 2011–2012) a data/.

17 379 registros horarios (hour.csv) y 731 diarios (day.csv): alquileres totales, de usuarios
casuales y registrados, con calendario y clima. Licencia CC BY 4.0.
Fuente: H. Fanaee-T, J. Gama, "Event labeling combining ensemble detectors and background
knowledge", Progress in AI (2013). https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Uso: uv run python 01-semestre-1/06-electiva-1/a-series-de-tiempo/data/download.py
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
WANTED = {"hour.csv", "day.csv"}


def main() -> None:
    if all((DATA_DIR / f).exists() for f in WANTED):
        print(f"Ya existe: {sorted(WANTED)} en {DATA_DIR}")
        return
    with urllib.request.urlopen(URL, timeout=120) as resp:
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if SHA256 is not None and digest != SHA256:
        raise RuntimeError(f"Checksum inesperado: {digest}")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name  # solo el nombre: evita rutas fuera de DATA_DIR
            if name in WANTED:
                (DATA_DIR / name).write_bytes(archive.read(member))
    print(f"Guardados {sorted(WANTED)} (sha256 del zip={digest})")


if __name__ == "__main__":
    main()
