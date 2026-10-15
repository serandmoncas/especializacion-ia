"""Solución ejercicio 2: minimax y alfa-beta en tres en raya.

Alfa-beta devuelve exactamente el mismo valor que minimax: solo deja de explorar ramas que no
pueden cambiar la decisión. Con ordenamiento perfecto visita O(b^{d/2}) nodos; un buen orden
heurístico (centro y esquinas primero) se acerca a ese ideal.
"""

import math

Board = tuple[str, ...]  # 9 casillas: "X", "O" o " "
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
GOOD_ORDER = [4, 0, 2, 6, 8, 1, 3, 5, 7]


def winner(b: Board) -> str | None:
    for i, j, k in LINES:
        if b[i] != " " and b[i] == b[j] == b[k]:
            return b[i]
    return None


def terminal_value(b: Board) -> int | None:
    """+1 si gana X, −1 si gana O, 0 si empate; None si el juego sigue."""
    w = winner(b)
    if w:
        return 1 if w == "X" else -1
    return 0 if " " not in b else None


def _moves(b: Board, ordered: bool) -> list[int]:
    order = GOOD_ORDER if ordered else range(9)
    return [i for i in order if b[i] == " "]


def _play(b: Board, i: int, player: str) -> Board:
    return (*b[:i], player, *b[i + 1 :])


def minimax(b: Board, player: str = "X") -> tuple[int, int]:
    """Devuelve (valor, nodos visitados). X maximiza, O minimiza."""
    v = terminal_value(b)
    if v is not None:
        return v, 1
    nodes, best = 1, -math.inf if player == "X" else math.inf
    for i in _moves(b, ordered=False):
        val, n = minimax(_play(b, i, player), "O" if player == "X" else "X")
        nodes += n
        best = max(best, val) if player == "X" else min(best, val)
    return int(best), nodes


def alphabeta(
    b: Board,
    player: str = "X",
    alpha: float = -math.inf,
    beta: float = math.inf,
    *,
    ordered: bool = False,
) -> tuple[int, int]:
    v = terminal_value(b)
    if v is not None:
        return v, 1
    nodes = 1
    if player == "X":
        best = -math.inf
        for i in _moves(b, ordered):
            val, n = alphabeta(_play(b, i, "X"), "O", alpha, beta, ordered=ordered)
            nodes += n
            best = max(best, val)
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # O nunca permitirá llegar aquí
    else:
        best = math.inf
        for i in _moves(b, ordered):
            val, n = alphabeta(_play(b, i, "O"), "X", alpha, beta, ordered=ordered)
            nodes += n
            best = min(best, val)
            beta = min(beta, best)
            if alpha >= beta:
                break
    return int(best), nodes


def check() -> None:
    empty: Board = (" ",) * 9
    v_mm, n_mm = minimax(empty)
    v_ab, n_ab = alphabeta(empty)
    v_ord, n_ord = alphabeta(empty, ordered=True)
    assert v_mm == v_ab == v_ord == 0, "el juego perfecto es empate"
    assert n_mm == 549_946, f"minimax visitó {n_mm} nodos (esperado 549 946)"
    assert n_ord < n_ab < n_mm
    x_wins: Board = ("X", "X", " ", "O", "O", " ", " ", " ", " ")
    assert alphabeta(x_wins, "X")[0] == 1
    print(f"ej02 ok — minimax {n_mm:,} nodos, alfa-beta {n_ab:,}, alfa-beta ordenado {n_ord:,}")


if __name__ == "__main__":
    check()
