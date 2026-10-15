"""Solución ejercicio 3: BFS bidireccional (costo unitario).

Se expande siempre la frontera más pequeña, una capa completa a la vez. Al terminar la capa en la
que las búsquedas se tocan, se elige el mejor punto de encuentro de esa capa: detenerse en el
primer contacto puede dar un camino una unidad más largo que el óptimo.

En grillas la ganancia es modesta: los estados a distancia ≤ d crecen como d² (no como bᵈ),
así que dos “discos” de radio d/2 cubren la mitad del área de uno de radio d, no una fracción
exponencialmente menor. Y en la peor dirección (esquina a esquina) ambas búsquedas acaban
llenando casi el mismo cuadrado. El ahorro O(b^{d/2}) es para espacios con crecimiento
exponencial, como los puzzles.
"""

from ai_foundations import breadth_first
from ai_foundations.problems.grid import GridMap, GridPathProblem


def bidirectional_bfs(grid: GridMap, start, goal) -> tuple[list | None, int]:
    """Devuelve (camino, nodos expandidos) en grilla 4-conectada."""
    if start == goal:
        return [start], 0
    parents = [{start: None}, {goal: None}]
    dist = [{start: 0}, {goal: 0}]
    frontiers = [[start], [goal]]
    expanded = 0
    while frontiers[0] and frontiers[1]:
        side = 0 if len(frontiers[0]) <= len(frontiers[1]) else 1
        other = 1 - side
        next_layer, best = [], None
        for cell in frontiers[side]:
            expanded += 1
            for nxt, _ in grid.neighbors(cell, diagonal=False):
                if nxt in parents[side]:
                    continue
                parents[side][nxt] = cell
                dist[side][nxt] = dist[side][cell] + 1
                next_layer.append(nxt)
                if nxt in parents[other]:
                    total = dist[side][nxt] + dist[other][nxt]
                    if best is None or total < best[0]:
                        best = (total, nxt)
        if best is not None:
            meet = best[1]
            left, node = [], meet
            while node is not None:
                left.append(node)
                node = parents[0][node]
            right, node = [], parents[1][meet]
            while node is not None:
                right.append(node)
                node = parents[1][node]
            return left[::-1] + right, expanded
        frontiers[side] = next_layer
    return None, expanded


def check() -> None:
    open_grid = GridMap.from_text("\n".join(["." * 60] * 60))
    for start, goal in [((0, 0), (59, 59)), ((5, 30), (55, 31)), ((10, 10), (10, 11))]:
        path, exp = bidirectional_bfs(open_grid, start, goal)
        ref = breadth_first(GridPathProblem(open_grid, start, goal, diagonal=False))
        assert len(path) - 1 == ref.cost, f"longitud {len(path) - 1} ≠ óptima {ref.cost}"
        assert path[0] == start and path[-1] == goal
        if ref.cost > 20:
            assert exp < ref.expanded
        print(f"{start}→{goal}: bidireccional {exp:>5} expandidos, BFS {ref.expanded:>5}")
    blocked = GridMap.from_text("..T..\n..T..\n..T..")
    assert bidirectional_bfs(blocked, (0, 0), (4, 0))[0] is None
    print("ej03 ok")


if __name__ == "__main__":
    check()
