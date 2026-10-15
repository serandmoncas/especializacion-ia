"""Descarga AL-CPL (Liang et al., 2018) con commit fijo y verifica el SHA-256 de cada archivo.

Pares de conceptos de Wikipedia (inglés) etiquetados como prerrequisito en cuatro dominios: data
mining, geometry, physics y precalculus. Repo: github.com/harrylclc/AL-CPL-dataset (sin licencia
explícita: se descarga para uso de estudio y no se redistribuye). Citar: Liang et al.,
«Investigating Active Learning for Concept Prerequisite Learning», EAAI 2018.

Wikipedia y OpenAlex no se descargan aquí: `run.py` los consulta bajo demanda y los guarda en
`data/cache/`.

Uso: uv run python 02-semestre-2/02-pln/data/download.py
"""

import hashlib
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
COMMIT = "11397eaa3ea01c5e06fefcfc1c02a84f21164081"
URL = "https://raw.githubusercontent.com/harrylclc/AL-CPL-dataset/{commit}/data/{name}"
SHA256 = {
    "data_mining.pairs": "9b0d42ac047f7e4b09f77210afcce5b4639f026c608d31f1d49e71c6b718e5da",
    "data_mining.preqs": "8dda7ec59e7199531baad22545d938acb840a20306ed79ae767e82dfb27d62de",
    "geometry.pairs": "883d6fa4622de6c2fc4581b835e3e1d3ab7d1ad7df00728c070897a0d09ec12c",
    "geometry.preqs": "e45787dfe417035c72a2a9cfa984ee7b75193d7b1f27e43a6023b363c9c69c40",
    "physics.pairs": "639070062f36c80e9393367f20f97e22b98ad3a36edbe36a5999ac930ddd13c4",
    "physics.preqs": "e1ce25603a7f3388ca2d7da034f7de7223eb40e62591415525687718fc059643",
    "precalculus.pairs": "20de87fbb394410c400871904d50cb723e5c88d31e12bdcc5919933478ba97cd",
    "precalculus.preqs": "80f8034f323d7853cd1036108292572115ebfdb52de78beeee493a5ca15b3015",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    dest = DATA_DIR / "alcpl"
    dest.mkdir(exist_ok=True)
    for name, digest in SHA256.items():
        path = dest / name
        if path.exists() and _sha256(path) == digest:
            continue
        with urllib.request.urlopen(URL.format(commit=COMMIT, name=name), timeout=60) as r:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(r.read())
        if _sha256(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"SHA-256 de {name} no coincide con el commit {COMMIT[:7]}")
        tmp.replace(path)
    print(f"AL-CPL listo en {dest} ({len(SHA256)} archivos verificados)")


if __name__ == "__main__":
    main()
