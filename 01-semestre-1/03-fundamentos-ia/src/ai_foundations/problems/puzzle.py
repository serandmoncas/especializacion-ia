"""Puzzle deslizante n×n (8-puzzle para n = 3, 15-puzzle para n = 4).

Estado: tupla de n² enteros en orden de filas; 0 es el hueco. Meta: (1, 2, …, n²−1, 0).
"""

from collections.abc import Iterator

import numpy as np

PuzzleState = tuple[int, ...]


def goal_state(size: int) -> PuzzleState:
    return (*range(1, size * size), 0)


def is_solvable(state: PuzzleState, size: int) -> bool:
    """Solo la mitad de las permutaciones es alcanzable desde la meta.

    Ancho impar: resoluble sii el número de inversiones es par.
    Ancho par: resoluble sii inversiones + (fila del hueco contada desde abajo, base 1) es impar.
    """
    tiles = [t for t in state if t != 0]
    inversions = sum(
        1 for i in range(len(tiles)) for j in range(i + 1, len(tiles)) if tiles[i] > tiles[j]
    )
    if size % 2 == 1:
        return inversions % 2 == 0
    blank_row_from_bottom = size - state.index(0) // size
    return (inversions + blank_row_from_bottom) % 2 == 1


class SlidingPuzzle:
    """Implementa `SearchProblem`. Cada movimiento cuesta 1."""

    def __init__(self, start: PuzzleState, size: int = 3) -> None:
        if sorted(start) != list(range(size * size)):
            raise ValueError("El estado debe ser una permutación de 0..n²−1")
        if not is_solvable(start, size):
            raise ValueError("Estado no resoluble")
        self.start, self.size, self.goal = start, size, goal_state(size)

    def initial_state(self) -> PuzzleState:
        return self.start

    def is_goal(self, state: PuzzleState) -> bool:
        return state == self.goal

    def successors(self, state: PuzzleState) -> Iterator[tuple[PuzzleState, float]]:
        n = self.size
        z = state.index(0)
        r, c = divmod(z, n)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                t = nr * n + nc
                s = list(state)
                s[z], s[t] = s[t], s[z]
                yield tuple(s), 1.0


def random_state(size: int, rng: np.random.Generator, *, walk: int | None = None) -> PuzzleState:
    """Estado aleatorio resoluble.

    Con `walk`: ese número de movimientos aleatorios desde la meta, sin deshacer el último.
    Sin `walk`: permutación uniforme entre las resolubles.
    """
    if walk is None:
        while True:
            state = tuple(int(v) for v in rng.permutation(size * size))
            if is_solvable(state, size):
                return state
    problem = SlidingPuzzle(goal_state(size), size)
    state, prev = problem.goal, None
    for _ in range(walk):
        options = [s for s, _ in problem.successors(state) if s != prev]
        prev, state = state, options[rng.integers(len(options))]
    return state


# --- Heurísticas -----------------------------------------------------------------------------


def misplaced(size: int):
    """h₁: fichas fuera de lugar (sin contar el hueco). Admisible: cada una debe moverse."""
    goal = goal_state(size)
    return lambda s: float(sum(1 for a, b in zip(s, goal, strict=True) if a != 0 and a != b))


def manhattan(size: int):
    """h₂: suma de distancias Manhattan de cada ficha a su lugar. Domina a h₁ (h₂ ≥ h₁)."""
    target = {t: divmod(t - 1, size) for t in range(1, size * size)}

    def h(s: PuzzleState) -> float:
        total = 0
        for i, t in enumerate(s):
            if t:
                r, c = divmod(i, size)
                tr, tc = target[t]
                total += abs(r - tr) + abs(c - tc)
        return float(total)

    return h


def _longest_increasing(seq: list[int]) -> int:
    tails: list[int] = []
    for v in seq:
        i = int(np.searchsorted(tails, v))
        tails[i : i + 1] = [v]
    return len(tails)


def linear_conflict(size: int):
    """h₃ = Manhattan + 2 por cada ficha que debe salir de su fila (o columna) meta y volver.

    Si en una línea hay fichas que pertenecen a ella pero en orden invertido, alguna debe salir
    de la línea: 2 movimientos que Manhattan no cuenta. El mínimo de fichas que deben salir es
    (fichas en la línea) − (subsecuencia creciente más larga de sus posiciones meta).
    Ojo: sumar 2 por cada *par* invertido sobreestima (3 2 1 tiene 3 pares pero basta sacar 2)
    y rompe la admisibilidad.
    """
    base = manhattan(size)
    target = {t: divmod(t - 1, size) for t in range(1, size * size)}

    def conflicts(line: list[tuple[int, int]]) -> int:
        # line: (posición actual, posición meta) de fichas cuya meta está en esta línea
        goals = [g for _, g in line]
        return len(goals) - _longest_increasing(goals)

    def h(s: PuzzleState) -> float:
        extra = 0
        for k in range(size):
            row = [
                (c, target[t][1])
                for c in range(size)
                if (t := s[k * size + c]) and target[t][0] == k
            ]
            col = [
                (r, target[t][0])
                for r in range(size)
                if (t := s[r * size + k]) and target[t][1] == k
            ]
            extra += conflicts(sorted(row)) + conflicts(sorted(col))
        return base(s) + 2 * extra

    return h
