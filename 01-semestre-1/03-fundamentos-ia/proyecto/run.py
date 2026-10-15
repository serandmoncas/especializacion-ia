"""Mini-proyecto: agente de búsqueda de caminos sobre mapas reales de Dragon Age: Origins.

Compara UCS, A* (octile), A* con heurística ALT (landmarks), Weighted A* y Greedy en los
benchmarks de Moving AI, y valida que los algoritmos óptimos reproducen la longitud óptima
publicada de cada escenario.

Uso:
    uv run python 01-semestre-1/03-fundamentos-ia/data/download.py
    uv run python 01-semestre-1/03-fundamentos-ia/proyecto/run.py   # ~1-2 min
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ai_foundations import astar, greedy_best_first, uniform_cost
from ai_foundations.landmarks import ALTHeuristic, select_landmarks
from ai_foundations.problems.grid import GridMap, GridPathProblem, Scenario, load_scenarios, octile
from ai_foundations.viz import plot_search

from common import data_dir, save_fig, set_seed, setup_style

MODULE = "01-semestre-1/03-fundamentos-ia"
OUT = Path(__file__).resolve().parent / "resultados"
SEED = 42
MAPS = ["den312d", "arena2", "ost003d", "brc202d"]
SCENARIOS_PER_MAP = 60
N_LANDMARKS = 8
OPTIMAL_ALGOS = {"ucs", "astar_octile", "astar_alt_random", "astar_alt_farthest"}


def sample_scenarios(scenarios: list[Scenario], k: int, rng: np.random.Generator) -> list[Scenario]:
    """Muestra estratificada por bucket (≈ longitud óptima / 4): cubre rutas cortas y largas."""
    buckets = sorted({s.bucket for s in scenarios})
    chosen = np.linspace(0, len(buckets) - 1, num=min(k, len(buckets))).round().astype(int)
    out = []
    for b in np.unique(np.array(buckets)[chosen]):
        pool = [s for s in scenarios if s.bucket == b]
        out.append(pool[rng.integers(len(pool))])
    return out


def build_algorithms(grid: GridMap, rng: np.random.Generator) -> tuple[dict, dict, dict]:
    """Devuelve ({nombre: f(problem, goal)}, segundos de precómputo, heurísticas ALT)."""
    algos = {
        "ucs": lambda p, goal: uniform_cost(p),
        "astar_octile": lambda p, goal: astar(p, octile(goal)),
        "wastar_1.5": lambda p, goal: astar(p, octile(goal), weight=1.5),
        "wastar_3": lambda p, goal: astar(p, octile(goal), weight=3.0),
        "greedy": lambda p, goal: greedy_best_first(p, octile(goal)),
    }
    prep, alts = {}, {}
    for strategy in ("random", "farthest"):
        try:
            t0 = time.perf_counter()
            landmarks = select_landmarks(grid, N_LANDMARKS, strategy=strategy, rng=rng)
            alt = ALTHeuristic(grid, landmarks)
            prep[f"astar_alt_{strategy}"] = time.perf_counter() - t0
        except NotImplementedError:
            continue
        alts[strategy] = alt
        algos[f"astar_alt_{strategy}"] = lambda p, goal, alt=alt: astar(p, alt.for_goal(goal))
    return algos, prep, alts


def run_map(name: str, rng: np.random.Generator) -> dict:
    root = data_dir(MODULE) / "dao"
    grid = GridMap.from_file(root / "maps" / f"{name}.map")
    scenarios = sample_scenarios(
        load_scenarios(root / "scen" / f"{name}.map.scen"), SCENARIOS_PER_MAP, rng
    )
    algos, prep, alts = build_algorithms(grid, rng)
    rows = []
    for sc in scenarios:
        problem = GridPathProblem(grid, sc.start, sc.goal)
        for algo, solve in algos.items():
            t0 = time.perf_counter()
            res = solve(problem, sc.goal)
            elapsed = time.perf_counter() - t0
            if algo in OPTIMAL_ALGOS and abs(res.cost - sc.optimal) > 1e-6:
                raise AssertionError(
                    f"{algo} en {name}: costo {res.cost} ≠ óptimo publicado {sc.optimal}"
                )
            rows.append(
                {
                    "algo": algo,
                    "bucket": sc.bucket,
                    "optimal": sc.optimal,
                    "cost": res.cost,
                    "ratio": res.cost / sc.optimal if sc.optimal > 0 else 1.0,
                    "expanded": res.expanded,
                    "seconds": elapsed,
                }
            )
    return {
        "grid": grid,
        "alts": alts,
        "rows": rows,
        "prep_seconds": prep,
        "n_scenarios": len(scenarios),
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for algo in dict.fromkeys(r["algo"] for r in rows):
        r = [x for x in rows if x["algo"] == algo]
        ucs = {(x["bucket"], x["optimal"]): x["expanded"] for x in rows if x["algo"] == "ucs"}
        reduction = [max(ucs[(x["bucket"], x["optimal"])], 1) / max(x["expanded"], 1) for x in r]
        out[algo] = {
            "median_expanded": float(np.median([x["expanded"] for x in r])),
            "geomean_reduction_vs_ucs": float(np.exp(np.mean(np.log(reduction)))),
            "mean_ratio": float(np.mean([x["ratio"] for x in r])),
            "max_ratio": float(np.max([x["ratio"] for x in r])),
            "total_seconds": float(np.sum([x["seconds"] for x in r])),
        }
    return out


COLORS = {
    "ucs": "C7",
    "astar_octile": "C0",
    "astar_alt_random": "C2",
    "astar_alt_farthest": "C4",
    "wastar_1.5": "C1",
    "wastar_3": "C3",
    "greedy": "C5",
}


def plot_expansions(all_rows: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in COLORS:
        r = [x for x in all_rows if x["algo"] == algo]
        if r:
            ax.scatter(
                [x["optimal"] for x in r],
                [x["expanded"] for x in r],
                s=8,
                alpha=0.5,
                color=COLORS[algo],
                label=algo,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("longitud óptima de la ruta")
    ax.set_ylabel("nodos expandidos")
    ax.set_title("Esfuerzo de búsqueda vs. longitud (4 mapas)")
    ax.legend(markerscale=2, fontsize=8)
    return fig


def plot_tradeoff(summary: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, s in summary.items():
        ax.scatter(s["geomean_reduction_vs_ucs"], s["mean_ratio"], s=60, color=COLORS[algo])
        ax.annotate(
            algo,
            (s["geomean_reduction_vs_ucs"], s["mean_ratio"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("reducción de expansiones vs. UCS (media geométrica, ×)")
    ax.set_ylabel("costo / óptimo (media)")
    ax.set_title("Velocidad vs. calidad")
    return fig


def plot_explored(grid: GridMap, sc: Scenario, alt: ALTHeuristic) -> plt.Figure:
    """Celdas expandidas (gris claro) y camino (rojo) de cuatro algoritmos en el mismo escenario."""
    problem = GridPathProblem(grid, sc.start, sc.goal)
    runs = {
        "ucs": uniform_cost(problem, record=True),
        "astar_octile": astar(problem, octile(sc.goal), record=True),
        "astar_alt_random": astar(problem, alt.for_goal(sc.goal), record=True),
        "wastar_3": astar(problem, octile(sc.goal), weight=3.0, record=True),
    }
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.5))
    for ax, (name, res) in zip(axes, runs.items(), strict=True):
        plot_search(ax, grid, res, name)
    return fig


def main() -> None:
    rng = set_seed(SEED)
    setup_style()
    OUT.mkdir(exist_ok=True)
    results, all_rows = {}, []
    for name in MAPS:
        t0 = time.perf_counter()
        r = run_map(name, rng)
        results[name] = r
        all_rows += r["rows"]
        print(f"{name}: {r['n_scenarios']} escenarios en {time.perf_counter() - t0:.1f}s")

    summary = summarize(all_rows)
    per_map = {name: summarize(r["rows"]) for name, r in results.items()}
    n_validated = sum(1 for x in all_rows if x["algo"] in OPTIMAL_ALGOS)
    report = {
        "maps": MAPS,
        "n_landmarks": N_LANDMARKS,
        "validated_optimal_runs": n_validated,
        "alt_preprocessing_seconds": {n: r["prep_seconds"] for n, r in results.items()},
        "summary": summary,
        "per_map": per_map,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    save_fig(plot_expansions(all_rows), OUT / "expansiones.png")
    save_fig(plot_tradeoff(summary), OUT / "velocidad_vs_calidad.png")
    grid = results["brc202d"]["grid"]
    scen = load_scenarios(data_dir(MODULE) / "dao" / "scen" / "brc202d.map.scen")
    hard = max((s for s in scen if s.bucket == 60), key=lambda s: s.optimal)
    save_fig(
        plot_explored(grid, hard, results["brc202d"]["alts"]["random"]),
        OUT / "exploracion_brc202d.png",
    )

    print(f"\n{n_validated} ejecuciones óptimas validadas contra la longitud publicada.")
    header = f"{'algoritmo':<20}{'mediana exp.':>13}{'× vs UCS':>10}{'costo/ópt':>11}"
    print(f"{header}{'máx':>8}{'seg':>8}")
    for algo, s in summary.items():
        print(
            f"{algo:<20}{s['median_expanded']:>13,.0f}{s['geomean_reduction_vs_ucs']:>10.1f}"
            f"{s['mean_ratio']:>11.4f}{s['max_ratio']:>8.3f}{s['total_seconds']:>8.1f}"
        )


if __name__ == "__main__":
    main()
