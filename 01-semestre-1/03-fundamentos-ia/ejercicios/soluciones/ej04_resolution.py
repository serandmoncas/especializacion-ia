"""Solución ejercicio 4: resolución proposicional.

Literales como strings ("P", "~P"); cláusulas como frozenset de literales. Resolver dos cláusulas
que contienen literales complementarios produce su unión sin ese par. La resolución es completa
por refutación: si KB ∧ ¬α es insatisfacible, se acabará derivando la cláusula vacía.
Las tautologías (contienen P y ~P) se descartan: nunca ayudan a refutar.
"""

import itertools

Clause = frozenset[str]


def negate(lit: str) -> str:
    return lit[1:] if lit.startswith("~") else "~" + lit


def resolve(a: Clause, b: Clause) -> list[Clause]:
    out = []
    for lit in a:
        if negate(lit) in b:
            resolvent = (a - {lit}) | (b - {negate(lit)})
            if not any(negate(x) in resolvent for x in resolvent):
                out.append(frozenset(resolvent))
    return out


def pl_resolution(kb: list[Clause], query_literal: str) -> bool:
    """KB ⊨ query_literal (un literal) por refutación."""
    clauses = set(kb) | {frozenset({negate(query_literal)})}
    while True:
        new: set[Clause] = set()
        for a, b in itertools.combinations(clauses, 2):
            for r in resolve(a, b):
                if not r:
                    return True
                new.add(r)
        if new <= clauses:
            return False
        clauses |= new


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
