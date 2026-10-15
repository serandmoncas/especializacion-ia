"""Descarga California Housing (StatLib, 20 640 filas) y la guarda como CSV en esta carpeta.

Uso: uv run python 01-semestre-1/01-matematicas-aplicadas/data/download.py
"""

from pathlib import Path

from sklearn.datasets import fetch_california_housing

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "california_housing.csv"


def main() -> None:
    if OUT.exists():
        print(f"Ya existe: {OUT}")
        return
    bunch = fetch_california_housing(data_home=DATA_DIR / ".cache", as_frame=True)
    bunch.frame.to_csv(OUT, index=False)
    print(f"Guardado {OUT} ({len(bunch.frame)} filas, {bunch.frame.shape[1]} columnas)")


if __name__ == "__main__":
    main()
