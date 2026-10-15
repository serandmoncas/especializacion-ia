"""Ejecuta notebooks de principio a fin y falla si alguna celda lanza error.

Por defecto solo los marcados como smoke (metadata del notebook: `"smoke": true`),
que deben correr en segundos y sin descargar datos. CI ejecuta este script.

Uso:
    uv run python scripts/run_notebooks.py            # smoke, sin modificar archivos
    uv run python scripts/run_notebooks.py --all      # todos
    uv run python scripts/run_notebooks.py --inplace  # guarda las salidas en el .ipynb
    uv run python scripts/run_notebooks.py ruta/a/notebook.ipynb
"""

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = ["00-preparatorio", "01-semestre-1", "02-semestre-2"]


def discover(include_all: bool) -> list[Path]:
    paths = sorted(
        p
        for d in SEARCH_DIRS
        for p in (ROOT / d).rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )
    if include_all:
        return paths
    return [p for p in paths if nbformat.read(p, as_version=4).metadata.get("smoke")]


def run(path: Path, *, inplace: bool, timeout: int) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": path.parent}}
    )
    start = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"FALLA  {path.relative_to(ROOT)}\n{exc}", file=sys.stderr)
        return False
    print(f"ok     {path.relative_to(ROOT)}  ({time.perf_counter() - start:.1f}s)")
    if inplace:
        nbformat.write(nb, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="incluir notebooks no-smoke")
    parser.add_argument("--inplace", action="store_true", help="guardar salidas")
    parser.add_argument("--timeout", type=int, default=600, help="segundos por celda")
    args = parser.parse_args()

    paths = [p.resolve() for p in args.paths] or discover(args.all)
    if not paths:
        print("No hay notebooks que ejecutar.")
        return 0
    results = [run(p, inplace=args.inplace, timeout=args.timeout) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
