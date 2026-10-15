"""Descarga CocoaMoniliaDataSet v2 (Zenodo 17716661, CC BY 4.0) y prepara fotos y recortes.

1 953 fotos de mazorcas de cacao (fincas de Santander y Antioquia) anotadas en cuatro etapas de
monilia: h0 sana, m1 jorobas, m2 manchas café, m3 esporulación. Cita: CocoaMoniliaDataSet, Data in
Brief (2025), DOI 10.1016/j.dib.2025.112447.

El zip pesa 6,2 GB: la descarga se reanuda si se corta y se verifica con el MD5 de Zenodo (y con el
SHA-256 fijado abajo). Solo se extraen las anotaciones COCO y las fotos.

Uso: uv run python 02-semestre-2/01-vision-por-computador/data/download.py [--borrar-zip]
"""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from applied_cv.cocoa.prepare import prepare

DATA_DIR = Path(__file__).resolve().parent
URL = "https://zenodo.org/records/17716661/files/{name}?download=1"
MD5 = {
    "CocoaMoniliaDataSet.zip": "d3531c4a62f3aaff9c146c7a8ae39a6b",
    "train.txt": "9b68d167efd0b55f6cb772b73eba0698",
    "val.txt": "b1ff099b8ec652a669d10deb0d9758eb",
    "test.txt": "6d9b3e6e3c7dbdbe01a86d18575f56db",
}
SHA256_ZIP = "fe5928b261e526cd259f87e282dad95094a29137b260e1f90bb3e2bdb2fef1e6"
KEEP = ("CocoaMoniliaDataSet/COCO_annotations/", "CocoaMoniliaDataSet/cocoapods_images/")


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest: Path) -> Path:
    """Descarga con reanudación (cabecera Range) y verifica el MD5 publicado por Zenodo."""
    path = dest / name
    if path.exists() and _digest(path, "md5") == MD5[name]:
        return path
    done = path.stat().st_size if path.exists() else 0
    req = urllib.request.Request(URL.format(name=name), headers={"Range": f"bytes={done}-"})
    with urllib.request.urlopen(req) as r, path.open("ab" if done else "wb") as f:
        if done and r.status != 206:  # el servidor no reanudó: empezar de cero
            f.truncate(0)
        shutil.copyfileobj(r, f, length=1 << 22)
    if _digest(path, "md5") != MD5[name]:
        raise RuntimeError(f"MD5 de {name} no coincide; borra el archivo y vuelve a intentar")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--borrar-zip", action="store_true", help="borrar el zip al terminar")
    args = parser.parse_args()
    for name in MD5:
        print(f"descargando {name}…", flush=True)
        download(name, DATA_DIR)
    zip_path = DATA_DIR / "CocoaMoniliaDataSet.zip"
    sha = _digest(zip_path, "sha256")
    if SHA256_ZIP is not None and sha != SHA256_ZIP:
        raise RuntimeError("SHA-256 del zip no coincide con el fijado")
    print(f"SHA-256 del zip: {sha}")
    raw = DATA_DIR / "raw"
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith(KEEP) and not m.endswith("/")]
        z.extractall(raw, members)
    # Las cajas se anotaron sobre la foto ya girada según EXIF (verificado con un mosaico: sin
    # aplicar EXIF, las mazorcas de 1 611 fotos con orientación 3, 6 u 8 quedan fuera de su caja).
    summary = prepare(raw / "CocoaMoniliaDataSet", DATA_DIR / "prepared", apply_exif=True)
    print({k: (v if not isinstance(v, list) else f"{len(v)}: {v[:5]}") for k, v in summary.items()})
    if args.borrar_zip:
        zip_path.unlink()
        print("zip borrado (6,2 GB liberados)")


if __name__ == "__main__":
    main()
