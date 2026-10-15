"""Ejercicio 1: N reinas como CSP (backtracking, MRV, forward checking). Ver README.md."""


def _attacks(r1: int, c1: int, r2: int, c2: int) -> bool:
    return r1 == r2 or abs(r1 - r2) == abs(c1 - c2)


def solve_naive(n: int, max_nodes: int = 2_000_000) -> tuple[list[int] | None, int]:
    """Backtracking cronológico columna por columna. Devuelve (filas por columna, nodos)."""
    raise NotImplementedError


def solve_mrv_fc(n: int) -> tuple[list[int] | None, int]:
    """Backtracking con MRV + forward checking."""
    raise NotImplementedError


def is_valid(rows: list[int]) -> bool:
    n = len(rows)
    return all(not _attacks(rows[i], i, rows[j], j) for i in range(n) for j in range(i + 1, n))


def check() -> None:
    for n in (8, 12, 20, 50):
        sol, nodes_fc = solve_mrv_fc(n)
        assert sol is not None and is_valid(sol), f"solución inválida para n={n}"
        naive, nodes_naive = solve_naive(n) if n <= 20 else (None, None)
        if naive is not None:
            assert is_valid(naive) and nodes_fc < nodes_naive
            naive_txt = f"{nodes_naive:>9,}"
        else:
            naive_txt = "abortado (> 2M)" if n <= 20 else "no intentado"
        print(f"n={n:>2}: ingenuo {naive_txt:>15} nodos | MRV+FC {nodes_fc:>4} nodos")
    print("ej01 ok")


if __name__ == "__main__":
    check()
