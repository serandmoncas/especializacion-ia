"""Descarga el experimento de email de Hillstrom (MineThatData, 2008) a data/hillstrom.csv.

64 000 clientes asignados al azar a 3 grupos (email de hombres, email de mujeres, sin email);
resultados a 2 semanas: visita, conversión y gasto. Es un experimento aleatorizado real,
ideal para A/B testing.

Uso: uv run python 01-semestre-1/02-estadistica-aplicada/data/download.py
"""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
OUT = Path(__file__).resolve().parent / "hillstrom.csv"
SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"


def main() -> None:
    if OUT.exists():
        print(f"Ya existe: {OUT}")
        return
    with urllib.request.urlopen(URL, timeout=60) as resp:
        content = resp.read()
    digest = hashlib.sha256(content).hexdigest()
    if SHA256 is not None and digest != SHA256:
        raise RuntimeError(f"Checksum inesperado: {digest}. ¿Cambió el archivo en origen?")
    OUT.write_bytes(content)
    print(f"Guardado {OUT} ({len(content) / 1e6:.1f} MB, sha256={digest[:12]}…)")


if __name__ == "__main__":
    main()
