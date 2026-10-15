import numpy as np
import pytest
from ai_foundations import astar, breadth_first, ida_star
from ai_foundations.problems.puzzle import (
    SlidingPuzzle,
    goal_state,
    is_solvable,
    linear_conflict,
    manhattan,
    misplaced,
    random_state,
)


def test_reachable_state_space_is_half_of_permutations(puzzle8_exact_distances):
    assert len(puzzle8_exact_distances) == 181_440  # 9!/2
    assert max(puzzle8_exact_distances.values()) == 31  # diámetro conocido del 8-puzzle


def test_solvability_matches_reachability(puzzle8_exact_distances):
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = tuple(int(v) for v in rng.permutation(9))
        assert is_solvable(s, 3) == (s in puzzle8_exact_distances)


@pytest.mark.parametrize("make_h", [misplaced, manhattan, linear_conflict])
def test_heuristics_are_admissible_on_every_state(make_h, puzzle8_exact_distances):
    h = make_h(3)
    violations = [s for s, d in puzzle8_exact_distances.items() if h(s) > d]
    assert not violations, f"{len(violations)} estados con h > h*, p. ej. {violations[0]}"


def test_dominance_chain(puzzle8_exact_distances):
    h1, h2, h3 = misplaced(3), manhattan(3), linear_conflict(3)
    states = list(puzzle8_exact_distances)[::97]
    assert all(h1(s) <= h2(s) <= h3(s) for s in states)
    assert any(h2(s) < h3(s) for s in states)


def test_linear_conflict_counts_tiles_not_pairs():
    # 3 2 1 en la fila superior: 3 pares invertidos pero basta sacar 2 fichas → +4, no +6.
    assert (
        linear_conflict(3)((3, 2, 1, 4, 5, 6, 7, 8, 0))
        == manhattan(3)((3, 2, 1, 4, 5, 6, 7, 8, 0)) + 4
    )


def test_fifteen_puzzle_solvability_rule():
    goal = goal_state(4)
    assert is_solvable(goal, 4)
    swapped = (*goal[:13], goal[14], goal[13], 0)  # intercambiar dos fichas: paridad impar
    assert not is_solvable(swapped, 4)


@pytest.mark.parametrize("seed", range(5))
def test_astar_and_ida_are_optimal(seed, puzzle8_exact_distances):
    rng = np.random.default_rng(seed)
    start = random_state(3, rng)
    problem = SlidingPuzzle(start, 3)
    optimal = puzzle8_exact_distances[start]
    assert astar(problem, manhattan(3)).cost == optimal
    assert ida_star(problem, linear_conflict(3)).cost == optimal


def test_better_heuristic_expands_fewer_nodes():
    start = random_state(3, np.random.default_rng(3), walk=40)
    problem = SlidingPuzzle(start, 3)
    e1 = astar(problem, misplaced(3)).expanded
    e2 = astar(problem, manhattan(3)).expanded
    e3 = astar(problem, linear_conflict(3)).expanded
    assert e1 > e2 >= e3


def test_bfs_solution_is_valid_sequence():
    start = random_state(3, np.random.default_rng(1), walk=12)
    problem = SlidingPuzzle(start, 3)
    res = breadth_first(problem)
    for a, b in zip(res.path, res.path[1:], strict=False):
        assert b in {s for s, _ in problem.successors(a)}
    assert res.path[-1] == goal_state(3)


def test_unsolvable_state_rejected():
    with pytest.raises(ValueError):
        SlidingPuzzle((2, 1, 3, 4, 5, 6, 7, 8, 0), 3)
