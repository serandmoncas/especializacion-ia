"""Visualización de búsquedas en grillas."""

import matplotlib.pyplot as plt
import numpy as np

from ai_foundations.problems.grid import GridMap
from ai_foundations.search import SearchResult


def plot_search(ax: plt.Axes, grid: GridMap, res: SearchResult, title: str = "") -> None:
    """Obstáculos en negro, celdas expandidas en gris (requiere `record=True`), camino en rojo."""
    img = np.where(grid.as_array(), 1.0, 0.0)
    for x, y in res.expanded_order:
        img[y, x] = 0.6
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if res.path:
        px, py = zip(*res.path, strict=True)
        ax.plot(px, py, color="C3", lw=1.5)
        ax.plot(px[0], py[0], "o", color="C2", ms=5)
        ax.plot(px[-1], py[-1], "*", color="C3", ms=9)
    cost = f"costo {res.cost:.1f}" if res.found else "sin camino"
    ax.set_title(f"{title}\n{res.expanded:,} expandidos, {cost}".strip(), fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
