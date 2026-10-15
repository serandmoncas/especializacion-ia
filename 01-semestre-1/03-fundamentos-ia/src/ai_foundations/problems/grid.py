"""Búsqueda de caminos en grillas con las reglas de los benchmarks de Moving AI.

- 8-conectividad: movimientos cardinales cuestan 1 y diagonales √2.
- Sin “cortar esquinas”: una diagonal solo es válida si las dos celdas cardinales que
  atraviesa también son transitables.
- Coordenadas (x, y) = (columna, fila), como en los archivos .map/.scen.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Cell = tuple[int, int]
SQRT2 = math.sqrt(2)
CARDINAL = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAGONAL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
PASSABLE = frozenset(".G")


class GridMap:
    def __init__(self, rows: list[str]) -> None:
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError("El mapa debe ser rectangular y no vacío")
        self.height, self.width = len(rows), len(rows[0])
        # Listas de Python: indexarlas es ~10× más rápido que un array de NumPy celda a celda.
        self._free = [[c in PASSABLE for c in row] for row in rows]

    @classmethod
    def from_text(cls, text: str) -> "GridMap":
        return cls([line for line in text.strip().splitlines()])

    @classmethod
    def from_file(cls, path: str | Path) -> "GridMap":
        """Lee el formato octile de Moving AI: cabecera `type/height/width/map` y luego filas."""
        lines = Path(path).read_text().splitlines()
        start = lines.index("map") + 1
        return cls(lines[start:])

    def as_array(self) -> np.ndarray:
        return np.array(self._free, dtype=bool)

    def passable(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and self._free[y][x]

    def neighbors(self, cell: Cell, *, diagonal: bool = True) -> Iterator[tuple[Cell, float]]:
        x, y = cell
        free = self.passable
        for dx, dy in CARDINAL:
            if free((x + dx, y + dy)):
                yield (x + dx, y + dy), 1.0
        if diagonal:
            for dx, dy in DIAGONAL:
                if free((x + dx, y + dy)) and free((x + dx, y)) and free((x, y + dy)):
                    yield (x + dx, y + dy), SQRT2

    def free_cells(self) -> list[Cell]:
        return [(x, y) for y in range(self.height) for x in range(self.width) if self._free[y][x]]


class GridPathProblem:
    """Ir de `start` a `goal` en un `GridMap`. Implementa el protocolo `SearchProblem`."""

    def __init__(self, grid: GridMap, start: Cell, goal: Cell, *, diagonal: bool = True) -> None:
        for name, cell in (("start", start), ("goal", goal)):
            if not grid.passable(cell):
                raise ValueError(f"{name} {cell} no es transitable")
        self.grid, self.start, self.goal, self.diagonal = grid, start, goal, diagonal

    def initial_state(self) -> Cell:
        return self.start

    def is_goal(self, state: Cell) -> bool:
        return state == self.goal

    def successors(self, state: Cell) -> Iterator[tuple[Cell, float]]:
        return self.grid.neighbors(state, diagonal=self.diagonal)


# --- Heurísticas: fábricas que fijan la meta y devuelven h(cell) ---------------------------


def manhattan(goal: Cell):
    """Admisible con 4-conectividad; NO admisible con diagonales (sobreestima)."""
    gx, gy = goal
    return lambda c: float(abs(c[0] - gx) + abs(c[1] - gy))


def euclidean(goal: Cell):
    """Admisible siempre, pero poco informada en grillas: subestima las rutas en ‘escalera’."""
    gx, gy = goal
    return lambda c: math.hypot(c[0] - gx, c[1] - gy)


def octile(goal: Cell):
    """Distancia exacta en grilla 8-conectada vacía: max + (√2 − 1)·min. Admisible y consistente."""
    gx, gy = goal

    def h(c: Cell) -> float:
        dx, dy = abs(c[0] - gx), abs(c[1] - gy)
        return max(dx, dy) + (SQRT2 - 1) * min(dx, dy)

    return h


# --- Escenarios de benchmark ----------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    bucket: int
    map_name: str
    start: Cell
    goal: Cell
    optimal: float


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Lee un .scen de Moving AI: bucket, mapa, ancho, alto, sx, sy, gx, gy, longitud óptima."""
    out = []
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 9:
            continue
        out.append(
            Scenario(int(f[0]), f[1], (int(f[4]), int(f[5])), (int(f[6]), int(f[7])), float(f[8]))
        )
    return out
