"""Descarga los benchmarks de búsqueda de caminos de Moving AI Lab (Dragon Age: Origins).

Mapas en formato octile (.map) y escenarios (.scen) con la longitud óptima publicada de cada
ruta, para validar A* contra una referencia externa.
Fuente: N. Sturtevant, "Benchmarks for Grid-Based Pathfinding", IEEE TCIAIG 4(2), 2012.
https://movingai.com/benchmarks/

Uso: uv run python 01-semestre-1/03-fundamentos-ia/data/download.py
"""

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
BASE = "https://movingai.com/benchmarks/dao/"
FILES = {
    "dao-map.zip": ("maps", "229393af31caaec732e5f4fd923940845ab056e94dadfb96b8afc24da472d3af"),
    "dao-scen.zip": ("scen", "7a0072450ff4c04fd4cf2c5e4651b11bfa4078582ecdcee95784e5196abf69ef"),
}


def safe_extract(archive: zipfile.ZipFile, dest: Path) -> int:
    """Extrae solo archivos regulares cuyo destino quede dentro de `dest` (evita zip-slip)."""
    dest = dest.resolve()
    count = 0
    for member in archive.infolist():
        if member.is_dir() or member.filename.startswith("__MACOSX"):
            continue
        target = (dest / Path(member.filename).name).resolve()
        if dest not in target.parents:
            raise RuntimeError(f"Ruta sospechosa en el zip: {member.filename}")
        target.write_bytes(archive.read(member))
        count += 1
    return count


def main() -> None:
    for name, (subdir, sha256) in FILES.items():
        dest = DATA_DIR / "dao" / subdir
        if dest.is_dir() and any(dest.iterdir()):
            print(f"Ya existe: {dest}")
            continue
        with urllib.request.urlopen(BASE + name, timeout=120) as resp:
            content = resp.read()
        digest = hashlib.sha256(content).hexdigest()
        if sha256 is not None and digest != sha256:
            raise RuntimeError(f"Checksum inesperado para {name}: {digest}")
        dest.mkdir(parents=True, exist_ok=True)
        n = safe_extract(zipfile.ZipFile(io.BytesIO(content)), dest)
        print(f"{name}: {n} archivos → {dest} (sha256={digest})")


if __name__ == "__main__":
    main()
