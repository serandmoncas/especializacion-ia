"""Heurística ALT (A*, Landmarks, Triangle inequality) para grafos con obstáculos.

Se precalcula la distancia exacta d(L, ·) desde unos pocos landmarks L. Por desigualdad
triangular, para todo n:  d(n, meta) ≥ |d(L, meta) − d(L, n)|. El máximo sobre landmarks es
admisible y consistente, y “conoce” los obstáculos, a diferencia de octile.

Funciona mejor cuando los landmarks quedan *detrás* de la meta o del inicio (en la periferia del
mapa): entonces la cota es casi exacta. La selección de landmarks es la decisión de diseño clave.
"""

import heapq
import math
from collections.abc import Callable

import numpy as np

from ai_foundations.problems.grid import Cell, GridMap


def dijkstra_map(grid: GridMap, source: Cell) -> np.ndarray:
    """Distancias exactas desde `source` a cada celda (inf si inalcanzable); array [alto, ancho]."""
    dist = np.full((grid.height, grid.width), math.inf)
    best = {source: 0.0}
    frontier = [(0.0, source)]
    while frontier:
        d, cell = heapq.heappop(frontier)
        if d > best[cell]:
            continue
        dist[cell[1], cell[0]] = d
        for nxt, cost in grid.neighbors(cell):
            nd = d + cost
            if nd < best.get(nxt, math.inf):
                best[nxt] = nd
                heapq.heappush(frontier, (nd, nxt))
    return dist


def select_landmarks(
    grid: GridMap, k: int, *, strategy: str = "random", rng: np.random.Generator | None = None
) -> list[Cell]:
    """Elige `k` landmarks entre las celdas transitables.

    - "random": uniforme. Línea base.
    - "farthest": selección por punto más lejano (ver TODO).
    """
    rng = rng or np.random.default_rng(0)
    cells = grid.free_cells()
    if strategy == "random":
        idx = rng.choice(len(cells), size=k, replace=False)
        return [cells[i] for i in idx]
    if strategy == "farthest":
        # TODO: implementar selección por punto más lejano (5-10 líneas). Ver README del proyecto.
        raise NotImplementedError
    raise ValueError(f"Estrategia desconocida: {strategy!r}")


class ALTHeuristic:
    """Precalcula d(L, ·) para cada landmark; `for_goal(goal)` devuelve h(cell)."""

    def __init__(self, grid: GridMap, landmarks: list[Cell]) -> None:
        self.landmarks = landmarks
        self._dist = np.stack([dijkstra_map(grid, lm) for lm in landmarks])  # [k, alto, ancho]
        # Listas anidadas por la misma razón que en GridMap: acceso escalar rápido.
        self._lists = [d.tolist() for d in self._dist]

    def for_goal(self, goal: Cell) -> Callable[[Cell], float]:
        gx, gy = goal
        tables = [(d, d[gy][gx]) for d in self._lists if d[gy][gx] != math.inf]

        def h(c: Cell) -> float:
            x, y = c
            return max((abs(dg - d[y][x]) for d, dg in tables), default=0.0)

        return h
