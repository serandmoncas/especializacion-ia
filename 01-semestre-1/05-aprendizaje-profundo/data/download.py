"""Descarga Fashion-MNIST (Xiao, Rasul & Vollgraf, 2017) a data/FashionMNIST/ vía torchvision.

70 000 imágenes 28×28 en escala de grises de 10 tipos de prenda (60 000 train / 10 000 test).
Licencia MIT. torchvision verifica el MD5 de cada archivo al descargarlo.

Uso: uv run python 01-semestre-1/05-aprendizaje-profundo/data/download.py
"""

from pathlib import Path

from torchvision.datasets import FashionMNIST

DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    for train in (True, False):
        ds = FashionMNIST(DATA_DIR, train=train, download=True)
        print(f"{'train' if train else 'test'}: {len(ds)} imágenes → {DATA_DIR / 'FashionMNIST'}")


if __name__ == "__main__":
    main()
