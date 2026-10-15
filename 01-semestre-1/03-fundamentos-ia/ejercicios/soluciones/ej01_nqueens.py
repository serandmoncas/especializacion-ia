"""Solución ejercicio 1: N reinas como CSP.

MRV ataca primero la variable más restringida: si va a fallar, falla pronto y poda una rama
grande. Forward checking detecta un dominio vacío antes de bajar un nivel. Juntos convierten
un problema exponencial en la práctica en casi lineal para N reinas.
"""


def _attacks(r1: int, c1: int, r2: int, c2: int) -> bool:
    return r1 == r2 or abs(r1 - r2) == abs(c1 - c2)


def solve_naive(n: int, max_nodes: int = 2_000_000) -> tuple[list[int] | None, int]:
    """Backtracking cronológico columna por columna. Devuelve (filas por columna, nodos)."""
    rows: list[int] = []
    nodes = 0

    def bt() -> bool:
        nonlocal nodes
        if len(rows) == n:
            return True
        c = len(rows)
        for r in range(n):
            nodes += 1
            if nodes > max_nodes:
                return False
            if all(not _attacks(r, c, rr, cc) for cc, rr in enumerate(rows)):
                rows.append(r)
                if bt():
                    return True
                rows.pop()
        return False

    return (list(rows) if bt() else None), nodes


def solve_mrv_fc(n: int) -> tuple[list[int] | None, int]:
    """Backtracking con MRV + forward checking."""
    domains = {c: set(range(n)) for c in range(n)}
    assignment: dict[int, int] = {}
    nodes = 0

    def bt() -> bool:
        nonlocal nodes
        if len(assignment) == n:
            return True
        col = min((c for c in domains if c not in assignment), key=lambda c: len(domains[c]))
        for r in sorted(domains[col]):
            nodes += 1
            pruned: list[tuple[int, int]] = []
            ok = True
            for c in domains:
                if c in assignment or c == col:
                    continue
                for rr in [rr for rr in domains[c] if _attacks(r, col, rr, c)]:
                    domains[c].discard(rr)
                    pruned.append((c, rr))
                if not domains[c]:
                    ok = False
                    break
            if ok:
                assignment[col] = r
                if bt():
                    return True
                del assignment[col]
            for c, rr in pruned:
                domains[c].add(rr)
        return False

    found = bt()
    return ([assignment[c] for c in range(n)] if found else None), nodes


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
