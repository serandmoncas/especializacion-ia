"""Estilo consistente para las figuras de notebooks y proyectos."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_style() -> None:
    """Estilo sobrio y legible; llamar una vez al inicio del notebook."""
    mpl.rcParams.update(
        {
            "figure.figsize": (7, 4.5),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, path: str | Path, *, dpi: int = 150) -> Path:
    """Guarda la figura creando carpetas intermedias; devuelve la ruta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path
