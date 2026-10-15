"""Rutas del repo. Los datos viven en `<módulo>/data/` (ignorado por git)."""

from pathlib import Path

_MARKER = "pyproject.toml"


def repo_root(start: Path | None = None) -> Path:
    """Sube desde `start` (o desde este archivo) hasta encontrar `pyproject.toml`."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / _MARKER).is_file():
            return candidate
    raise FileNotFoundError(f"No se encontró {_MARKER} por encima de {here}")


def module_dir(relative: str) -> Path:
    """Ruta absoluta de un módulo, p. ej. `module_dir("01-semestre-1/01-matematicas-aplicadas")`."""
    path = repo_root() / relative
    if not path.is_dir():
        raise FileNotFoundError(f"El módulo no existe: {path}")
    return path


def data_dir(relative: str, *, create: bool = True) -> Path:
    """Carpeta `data/` de un módulo; se crea si no existe."""
    path = module_dir(relative) / "data"
    if create:
        path.mkdir(exist_ok=True)
    return path
