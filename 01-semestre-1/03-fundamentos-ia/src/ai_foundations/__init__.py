"""Fundamentos de IA: agentes, búsqueda (no informada e informada) y lógica proposicional."""

from ai_foundations.search import (
    SearchProblem,
    SearchResult,
    astar,
    best_first,
    breadth_first,
    effective_branching_factor,
    greedy_best_first,
    ida_star,
    iterative_deepening,
    uniform_cost,
)

__all__ = [
    "SearchProblem",
    "SearchResult",
    "astar",
    "best_first",
    "breadth_first",
    "effective_branching_factor",
    "greedy_best_first",
    "ida_star",
    "iterative_deepening",
    "uniform_cost",
]
