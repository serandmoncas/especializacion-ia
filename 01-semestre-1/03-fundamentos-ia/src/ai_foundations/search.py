"""Algoritmos de búsqueda sobre un problema genérico.

Un problema expone tres cosas (protocolo `SearchProblem`): estado inicial, test de meta y
sucesores con su costo. Los algoritmos no saben nada del dominio; la heurística se pasa aparte.

Todos devuelven `SearchResult` con métricas comparables:
- expanded: nodos sacados de la frontera y expandidos (el costo real de la búsqueda).
- generated: sucesores generados.
- max_frontier: tamaño máximo de la frontera (proxy de memoria).
"""

import heapq
import itertools
import math
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

State = Hashable
Heuristic = Callable[[State], float]


class SearchProblem(Protocol):
    def initial_state(self) -> State: ...
    def is_goal(self, state: State) -> bool: ...
    def successors(self, state: State) -> Iterable[tuple[State, float]]: ...


@dataclass
class SearchResult:
    path: list[State] | None  # de inicial a meta, ambos incluidos; None si no hay solución
    cost: float
    expanded: int
    generated: int
    max_frontier: int
    expanded_order: list[State] = field(default_factory=list, repr=False)  # solo si record=True

    @property
    def found(self) -> bool:
        return self.path is not None


def _reconstruct(parents: dict[State, State | None], goal: State) -> list[State]:
    path = [goal]
    while (parent := parents[path[-1]]) is not None:
        path.append(parent)
    return path[::-1]


def breadth_first(problem: SearchProblem, *, record: bool = False) -> SearchResult:
    """BFS en grafo. Óptima solo si todos los costos son iguales: minimiza el número de pasos.

    El test de meta se hace al *generar* (no al expandir): con costos unitarios es seguro
    y ahorra expandir una capa entera.
    """
    start = problem.initial_state()
    if problem.is_goal(start):
        return SearchResult([start], 0.0, 0, 0, 1)
    parents: dict[State, State | None] = {start: None}
    frontier = deque([start])
    expanded = generated = 0
    max_frontier = 1
    order: list[State] = []
    while frontier:
        state = frontier.popleft()
        expanded += 1
        if record:
            order.append(state)
        for nxt, _ in problem.successors(state):
            generated += 1
            if nxt in parents:
                continue
            parents[nxt] = state
            if problem.is_goal(nxt):
                path = _reconstruct(parents, nxt)
                return SearchResult(
                    path, float(len(path) - 1), expanded, generated, max_frontier, order
                )
            frontier.append(nxt)
        max_frontier = max(max_frontier, len(frontier))
    return SearchResult(None, math.inf, expanded, generated, max_frontier, order)


def iterative_deepening(problem: SearchProblem, *, max_depth: int = 50) -> SearchResult:
    """DFS con límite creciente: memoria O(b·d) de DFS, completitud y optimalidad (en pasos) de BFS.

    Re-expande los niveles superiores en cada iteración, pero con factor de ramificación b el
    sobrecosto es solo ~ b/(b−1): la última capa domina.
    """
    expanded = generated = 0

    def dls(path: list[State], on_path: set[State], limit: int) -> list[State] | None:
        nonlocal expanded, generated
        state = path[-1]
        if problem.is_goal(state):
            return list(path)
        if limit == 0:
            return None
        expanded += 1
        for nxt, _ in problem.successors(state):
            generated += 1
            if nxt in on_path:  # evita ciclos sobre el camino actual
                continue
            path.append(nxt)
            on_path.add(nxt)
            found = dls(path, on_path, limit - 1)
            path.pop()
            on_path.discard(nxt)
            if found is not None:
                return found
        return None

    start = problem.initial_state()
    for depth in range(max_depth + 1):
        found = dls([start], {start}, depth)
        if found is not None:
            return SearchResult(found, float(len(found) - 1), expanded, generated, depth + 1)
    return SearchResult(None, math.inf, expanded, generated, max_depth + 1)


# Mejoras de g menores que esto se consideran ruido de redondeo: sumar √2 en distinto orden da
# g que difieren en ~1e-15, y sin tolerancia A* “reabre” estados por caminos idénticos.
G_TOLERANCE = 1e-9


def best_first(
    problem: SearchProblem,
    priority: Callable[[float, float], float],
    h: Heuristic,
    *,
    reopen: bool = True,
    record: bool = False,
) -> SearchResult:
    """Búsqueda primero-el-mejor genérica con f = priority(g, h).

    - UCS:     f = g
    - Greedy:  f = h
    - A*:      f = g + h
    - WA*:     f = g + w·h

    Empates en f se rompen por menor h (= mayor g, más cerca de la meta), lo que en grillas con
    muchos caminos equivalentes reduce mucho las expansiones.

    `reopen`: si aparece un camino más barato a un estado ya expandido, ¿volver a expandirlo?
    Necesario para que A* sea óptima con heurísticas admisibles pero inconsistentes. Para WA*
    y Greedy es contraproducente: re-expanden el mismo estado decenas de veces, y WA* conserva su
    cota de costo ≤ w·óptimo sin reabrir (Likhachev et al., ARA*, 2003).
    """
    start = problem.initial_state()
    counter = itertools.count()
    h0 = h(start)
    frontier: list[tuple[float, float, int, State]] = [
        (priority(0.0, h0), h0, next(counter), start)
    ]
    g: dict[State, float] = {start: 0.0}
    parents: dict[State, State | None] = {start: None}
    expanded = generated = 0
    max_frontier = 1
    order: list[State] = []
    closed: set[State] = set()
    while frontier:
        _, _, _, state = heapq.heappop(frontier)
        if state in closed:
            continue  # entrada obsoleta: ya se expandió con un g mejor o igual
        if problem.is_goal(state):
            return SearchResult(
                _reconstruct(parents, state), g[state], expanded, generated, max_frontier, order
            )
        closed.add(state)
        expanded += 1
        if record:
            order.append(state)
        g_state = g[state]
        for nxt, cost in problem.successors(state):
            generated += 1
            if not reopen and nxt in closed:
                continue
            g_new = g_state + cost
            if g_new < g.get(nxt, math.inf) - G_TOLERANCE:
                g[nxt] = g_new
                parents[nxt] = state
                closed.discard(nxt)  # reapertura (solo ocurre si reopen=True)
                h_nxt = h(nxt)
                heapq.heappush(frontier, (priority(g_new, h_nxt), h_nxt, next(counter), nxt))
        max_frontier = max(max_frontier, len(frontier))
    return SearchResult(None, math.inf, expanded, generated, max_frontier, order)


def _zero(_: State) -> float:
    return 0.0


def uniform_cost(problem: SearchProblem, *, record: bool = False) -> SearchResult:
    """UCS (Dijkstra con meta): óptima para costos no negativos."""
    return best_first(problem, lambda g, h: g, _zero, record=record)


def greedy_best_first(
    problem: SearchProblem, h: Heuristic, *, record: bool = False
) -> SearchResult:
    """Greedy: solo mira h. Rápida, sin garantía de optimalidad. Nunca reabre: no le importa g."""
    return best_first(problem, lambda g, h: h, h, reopen=False, record=record)


def astar(
    problem: SearchProblem, h: Heuristic, *, weight: float = 1.0, record: bool = False
) -> SearchResult:
    """A* (weight = 1) o Weighted A* (weight > 1).

    Con h admisible, A* es óptima. WA* garantiza costo ≤ weight · óptimo y suele expandir
    muchos menos nodos: es la perilla entre calidad y velocidad. WA* no reabre estados.
    """
    if weight == 1.0:
        return best_first(problem, lambda g, h: g + h, h, record=record)
    return best_first(problem, lambda g, h: g + weight * h, h, reopen=False, record=record)


def ida_star(problem: SearchProblem, h: Heuristic, *, max_iterations: int = 1000) -> SearchResult:
    """IDA*: profundización iterativa sobre el umbral f = g + h. Memoria lineal en la profundidad.

    Cada iteración sube el umbral al menor f que lo excedió. Ideal en puzzles (8/15-puzzle)
    con costos unitarios; mala en grillas, donde re-expande muchísimos caminos equivalentes.
    """
    start = problem.initial_state()
    expanded = generated = 0
    path, on_path = [start], {start}

    def search(g_cost: float, bound: float) -> tuple[bool, float]:
        nonlocal expanded, generated
        state = path[-1]
        f = g_cost + h(state)
        if f > bound:
            return False, f
        if problem.is_goal(state):
            return True, g_cost
        expanded += 1
        minimum = math.inf
        for nxt, cost in problem.successors(state):
            generated += 1
            if nxt in on_path:
                continue
            path.append(nxt)
            on_path.add(nxt)
            found, value = search(g_cost + cost, bound)
            if found:
                return True, value
            path.pop()
            on_path.discard(nxt)
            minimum = min(minimum, value)
        return False, minimum

    bound = h(start)
    for _ in range(max_iterations):
        found, value = search(0.0, bound)
        if found:
            return SearchResult(list(path), value, expanded, generated, len(path))
        if value == math.inf:
            break
        bound = value
    return SearchResult(None, math.inf, expanded, generated, 0)


def effective_branching_factor(expanded: int, depth: int, *, tol: float = 1e-6) -> float:
    """b* tal que 1 + b* + b*² + … + b*^d = N + 1 (Russell & Norvig).

    Resume la calidad de una heurística en un número casi independiente de la profundidad:
    cuanto más cerca de 1, mejor.
    """
    if depth == 0:
        return 0.0
    target = expanded + 1

    def total(b: float) -> float:
        return float(depth + 1) if b == 1 else (b ** (depth + 1) - 1) / (b - 1)

    lo, hi = 1.0, max(2.0, float(expanded))
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
