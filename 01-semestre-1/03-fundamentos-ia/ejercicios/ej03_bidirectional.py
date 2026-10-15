"""Ejercicio 3: BFS bidireccional en grillas. Ver README.md."""

from ai_foundations import breadth_first
from ai_foundations.problems.grid import GridMap, GridPathProblem


def bidirectional_bfs(grid: GridMap, start, goal) -> tuple[list | None, int]:
    """Devuelve (camino, nodos expandidos) en grilla 4-conectada."""
    raise NotImplementedError


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
