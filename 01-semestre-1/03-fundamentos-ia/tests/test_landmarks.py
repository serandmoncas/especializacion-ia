import math

import numpy as np
import pytest
from ai_foundations import astar
from ai_foundations.landmarks import ALTHeuristic, dijkstra_map, select_landmarks
from ai_foundations.problems.grid import GridMap, GridPathProblem, octile

MAZE = GridMap.from_text(
    """
..........T.........
.TTTTTTTT.T.TTTTTTT.
.T......T.T.T.....T.
.T.TTTT.T...T.TTT.T.
.T.T..T.TTTTT.T.T.T.
.T.T..T.......T.T.T.
.T.TTTTTTTTTTTT.T.T.
.T..............T.T.
.TTTTTTTTTTTTTTTT.T.
....................
"""
)


def test_dijkstra_map_matches_astar():
    src = (0, 0)
    dist = dijkstra_map(MAZE, src)
    for goal in [(19, 9), (2, 7), (11, 3)]:
        res = astar(GridPathProblem(MAZE, src, goal), octile(goal))
        assert dist[goal[1], goal[0]] == pytest.approx(res.cost)
    assert math.isinf(dist[0, 10])  # árbol
    assert math.isinf(dist[4, 4])  # celda libre pero encerrada


def test_alt_is_admissible_and_beats_octile():
    rng = np.random.default_rng(0)
    alt = ALTHeuristic(MAZE, select_landmarks(MAZE, 4, rng=rng))
    goal = (2, 7)  # a 2.8 en línea recta de (0, 9) pero a 50 rodeando el laberinto
    true = dijkstra_map(MAZE, goal)  # grafo no dirigido: d(n, goal) = d(goal, n)
    h = alt.for_goal(goal)
    for cell in MAZE.free_cells():
        if math.isfinite(true[cell[1], cell[0]]):
            assert h(cell) <= true[cell[1], cell[0]] + 1e-9
    p = GridPathProblem(MAZE, (0, 9), goal)
    with_alt = astar(p, h)
    with_octile = astar(p, octile(goal))
    assert with_alt.cost == pytest.approx(with_octile.cost)
    assert with_alt.expanded < with_octile.expanded


def test_landmark_at_goal_gives_exact_heuristic():
    goal = (2, 7)
    h = ALTHeuristic(MAZE, [goal]).for_goal(goal)
    true = dijkstra_map(MAZE, goal)
    for cell in MAZE.free_cells():
        if math.isfinite(true[cell[1], cell[0]]):
            assert h(cell) == pytest.approx(true[cell[1], cell[0]])


def test_select_landmarks_random_is_valid_and_reproducible():
    a = select_landmarks(MAZE, 5, rng=np.random.default_rng(1))
    b = select_landmarks(MAZE, 5, rng=np.random.default_rng(1))
    assert a == b and len(set(a)) == 5
    assert all(MAZE.passable(c) for c in a)


def test_unknown_strategy():
    with pytest.raises(ValueError):
        select_landmarks(MAZE, 2, strategy="magic")
