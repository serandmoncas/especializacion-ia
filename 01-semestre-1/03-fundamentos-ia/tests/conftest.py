from collections import deque

import pytest
from ai_foundations.problems.puzzle import SlidingPuzzle, goal_state


@pytest.fixture(scope="session")
def puzzle8_exact_distances() -> dict[tuple[int, ...], int]:
    """h* exacta de los 181 440 estados del 8-puzzle por BFS hacia atrás desde la meta."""
    goal = goal_state(3)
    problem = SlidingPuzzle(goal, 3)
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        s = queue.popleft()
        for nxt, _ in problem.successors(
            s
        ):  # las acciones son reversibles: el grafo es no dirigido
            if nxt not in dist:
                dist[nxt] = dist[s] + 1
                queue.append(nxt)
    return dist
