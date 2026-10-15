"""Descarga UCI Adult (Census Income, 1994) a data/: adult.data (train) y adult.test (test).

48 842 personas, 14 atributos (edad, educación, ocupación, horas, …), objetivo: ingreso > 50K USD.
Particiones oficiales: 32 561 train / 16 281 test. Licencia CC BY 4.0.
Fuente: B. Becker, R. Kohavi, https://archive.ics.uci.edu/dataset/2/adult

Uso: uv run python 01-semestre-1/04-aprendizaje-automatico/data/download.py
"""

import hashlib
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILES = {
    "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
}


def main() -> None:
    for name, sha256 in FILES.items():
        out = DATA_DIR / name
        if out.exists():
            print(f"Ya existe: {out}")
            continue
        with urllib.request.urlopen(BASE + name, timeout=120) as resp:
            content = resp.read()
        digest = hashlib.sha256(content).hexdigest()
        if sha256 is not None and digest != sha256:
            raise RuntimeError(f"Checksum inesperado para {name}: {digest}")
        out.write_bytes(content)
        print(f"{name}: {len(content) / 1e6:.1f} MB (sha256={digest})")


if __name__ == "__main__":
    main()
