import math

import numpy as np
import pytest
from ai_foundations import (
    astar,
    breadth_first,
    effective_branching_factor,
    greedy_best_first,
    iterative_deepening,
    uniform_cost,
)
from ai_foundations.problems.grid import (
    SQRT2,
    GridMap,
    GridPathProblem,
    euclidean,
    load_scenarios,
    manhattan,
    octile,
)

OPEN = GridMap.from_text("\n".join(["." * 12] * 10))
WALL = GridMap.from_text(
    """
............
............
..TTTTTTTT..
.........T..
.........T..
..TTTTTT.T..
.........T..
............
"""
)


def random_grid(seed: int, size: int = 30, density: float = 0.25) -> GridMap:
    rng = np.random.default_rng(seed)
    blocked = rng.random((size, size)) < density
    return GridMap(["".join("T" if b else "." for b in row) for row in blocked])


def test_no_corner_cutting():
    g = GridMap.from_text("..\nT.")
    neighbors = dict(g.neighbors((0, 0)))
    assert (1, 1) not in neighbors  # la diagonal rozaría el árbol en (0, 1)
    assert neighbors == {(1, 0): 1.0}


def test_diagonal_cost_and_octile_is_exact_on_open_grid():
    for goal in [(11, 9), (5, 0), (0, 7), (11, 3)]:
        res = astar(GridPathProblem(OPEN, (0, 0), goal), octile(goal))
        assert res.cost == pytest.approx(octile(goal)((0, 0)))


def test_astar_matches_ucs_on_random_maps():
    for seed in range(8):
        g = random_grid(seed)
        free = g.free_cells()
        rng = np.random.default_rng(seed)
        s, t = (free[i] for i in rng.choice(len(free), 2, replace=False))
        p = GridPathProblem(g, s, t)
        ucs, a = uniform_cost(p), astar(p, octile(t))
        assert ucs.found == a.found
        if ucs.found:
            assert a.cost == pytest.approx(ucs.cost)
            assert a.expanded <= ucs.expanded


@pytest.mark.parametrize("w", [1.5, 2.0, 5.0])
def test_weighted_astar_respects_bound(w):
    for seed in range(8):
        g = random_grid(seed, density=0.3)
        free = g.free_cells()
        rng = np.random.default_rng(100 + seed)
        s, t = (free[i] for i in rng.choice(len(free), 2, replace=False))
        p = GridPathProblem(g, s, t)
        opt = astar(p, octile(t))
        if opt.found:
            assert astar(p, octile(t), weight=w).cost <= w * opt.cost + 1e-9


def test_greedy_is_never_better_and_sometimes_worse():
    gaps = []
    for seed in range(20):
        g = random_grid(seed, density=0.3)
        free = g.free_cells()
        rng = np.random.default_rng(200 + seed)
        s, t = (free[i] for i in rng.choice(len(free), 2, replace=False))
        p = GridPathProblem(g, s, t)
        opt = astar(p, octile(t))
        if opt.found:
            gaps.append(greedy_best_first(p, octile(t)).cost - opt.cost)
    assert min(gaps) >= -1e-9
    assert max(gaps) > 0


def test_manhattan_is_inadmissible_with_diagonals():
    goal = (5, 5)
    h = manhattan(goal)
    true_cost = astar(GridPathProblem(OPEN, (0, 0), goal), octile(goal)).cost
    assert h((0, 0)) > true_cost  # 10 > 5√2: sobreestima


def test_heuristic_ordering_on_open_grid():
    goal = (11, 9)
    cell = (0, 0)
    assert euclidean(goal)(cell) <= octile(goal)(cell) <= manhattan(goal)(cell)


def test_bfs_and_iddfs_minimize_steps_with_4_connectivity():
    p = GridPathProblem(WALL, (0, 0), (11, 7), diagonal=False)
    bfs, ids = breadth_first(p), iterative_deepening(p, max_depth=40)
    ucs = uniform_cost(GridPathProblem(WALL, (0, 0), (11, 7), diagonal=False))
    assert bfs.cost == ids.cost == ucs.cost == 18


def test_unreachable_goal():
    g = GridMap.from_text(".T.\n.T.\n.T.")
    res = astar(GridPathProblem(g, (0, 0), (2, 2)), octile((2, 2)))
    assert not res.found and res.cost == math.inf


def test_blocked_endpoint_rejected():
    with pytest.raises(ValueError):
        GridPathProblem(WALL, (2, 2), (0, 0))


def test_record_expanded_order():
    res = astar(GridPathProblem(OPEN, (0, 0), (3, 3)), octile((3, 3)), record=True)
    assert res.expanded_order[0] == (0, 0)
    assert len(res.expanded_order) == res.expanded


def test_effective_branching_factor():
    assert effective_branching_factor(14, 3) == pytest.approx(2.0, abs=1e-4)  # 2 + 4 + 8
    assert effective_branching_factor(3, 3) == pytest.approx(1.0, abs=1e-4)


def test_load_scenarios(tmp_path):
    f = tmp_path / "x.map.scen"
    f.write_text("version 1\n3\tx.map\t49\t49\t1\t2\t3\t4\t2.41421356\n")
    (sc,) = load_scenarios(f)
    assert sc.start == (1, 2) and sc.goal == (3, 4)
    assert sc.optimal == pytest.approx(1 + SQRT2)


def test_load_moving_ai_map(tmp_path):
    f = tmp_path / "m.map"
    f.write_text("type octile\nheight 2\nwidth 3\nmap\n..@\nT..\n")
    g = GridMap.from_file(f)
    assert (g.width, g.height) == (3, 2)
    assert g.passable((0, 0)) and not g.passable((2, 0)) and not g.passable((0, 1))


def test_astar_with_consistent_heuristic_never_reexpands():
    # Regresión: sin tolerancia en g, el redondeo de sumas de √2 causaba re-expansiones.
    for seed in range(5):
        g = random_grid(seed, size=40, density=0.2)
        free = g.free_cells()
        rng = np.random.default_rng(300 + seed)
        s, t = (free[i] for i in rng.choice(len(free), 2, replace=False))
        res = astar(GridPathProblem(g, s, t), octile(t), record=True)
        assert len(res.expanded_order) == len(set(res.expanded_order))


@pytest.mark.parametrize("w", [1.5, 3.0])
def test_weighted_astar_never_reexpands(w):
    g = random_grid(1, size=40, density=0.3)
    free = g.free_cells()
    s, t = free[0], free[-1]
    res = astar(GridPathProblem(g, s, t), octile(t), weight=w, record=True)
    assert len(res.expanded_order) == len(set(res.expanded_order))
