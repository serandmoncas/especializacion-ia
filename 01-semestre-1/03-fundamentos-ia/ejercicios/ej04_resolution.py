"""Ejercicio 4: resolución proposicional por refutación. Ver README.md."""

Clause = frozenset[str]


def negate(lit: str) -> str:
    return lit[1:] if lit.startswith("~") else "~" + lit


def resolve(a: Clause, b: Clause) -> list[Clause]:
    raise NotImplementedError


def pl_resolution(kb: list[Clause], query_literal: str) -> bool:
    """KB ⊨ query_literal (un literal) por refutación."""
    raise NotImplementedError


def check() -> None:
    from ai_foundations.logic import conjoin, iff, symbols, tt_entails

    # Wumpus del notebook 04 en CNF:
    # B11 ⇔ (P12 ∨ P21), B21 ⇔ (P12 ∨ P22 ∨ P31), ¬B11, B21
    kb = [
        frozenset(c)
        for c in [
            {"~B11", "P12", "P21"},
            {"~P12", "B11"},
            {"~P21", "B11"},
            {"~B21", "P12", "P22", "P31"},
            {"~P12", "B21"},
            {"~P22", "B21"},
            {"~P31", "B21"},
            {"~B11"},
            {"B21"},
        ]
    ]
    B11, B21, P12, P21, P22, P31 = symbols("B11 B21 P12 P21 P22 P31")
    kb_expr = conjoin([iff(B11, P12 | P21), iff(B21, P12 | P22 | P31), ~B11, B21])
    for lit, expr in [("~P12", ~P12), ("~P21", ~P21), ("P22", P22), ("~P22", ~P22), ("B21", B21)]:
        assert pl_resolution(kb, lit) == tt_entails(kb_expr, expr), lit
    assert pl_resolution(kb, "~P12") and not pl_resolution(kb, "P22")
    print("ej04 ok — resolución coincide con enumeración de modelos")


if __name__ == "__main__":
    check()
