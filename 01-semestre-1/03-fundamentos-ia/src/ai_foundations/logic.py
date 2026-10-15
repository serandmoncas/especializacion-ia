"""Lógica proposicional: representación, model checking y encadenamiento hacia adelante.

Las fórmulas se construyen con operadores de Python:
    ~A (no), A & B (y), A | B (o), A >> B (implica), iff(A, B) (si y solo si).
"""

import itertools
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass


class Expr:
    def __invert__(self) -> "Expr":
        return Not(self)

    def __and__(self, other: "Expr") -> "Expr":
        return And(self, other)

    def __or__(self, other: "Expr") -> "Expr":
        return Or(self, other)

    def __rshift__(self, other: "Expr") -> "Expr":
        return Implies(self, other)


@dataclass(frozen=True)
class Symbol(Expr):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not(Expr):
    arg: Expr

    def __repr__(self) -> str:
        return f"¬{self.arg!r}"


@dataclass(frozen=True)
class And(Expr):
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left!r} ∧ {self.right!r})"


@dataclass(frozen=True)
class Or(Expr):
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left!r} ∨ {self.right!r})"


@dataclass(frozen=True)
class Implies(Expr):
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left!r} ⇒ {self.right!r})"


@dataclass(frozen=True)
class Iff(Expr):
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left!r} ⇔ {self.right!r})"


def iff(a: Expr, b: Expr) -> Expr:
    return Iff(a, b)


def symbols(names: str) -> tuple[Symbol, ...]:
    return tuple(Symbol(n) for n in names.split())


def conjoin(exprs: Iterable[Expr]) -> Expr:
    exprs = list(exprs)
    out = exprs[0]
    for e in exprs[1:]:
        out = And(out, e)
    return out


def evaluate(e: Expr, model: dict[Symbol, bool]) -> bool:
    match e:
        case Symbol():
            return model[e]
        case Not(a):
            return not evaluate(a, model)
        case And(a, b):
            return evaluate(a, model) and evaluate(b, model)
        case Or(a, b):
            return evaluate(a, model) or evaluate(b, model)
        case Implies(a, b):
            return (not evaluate(a, model)) or evaluate(b, model)
        case Iff(a, b):
            return evaluate(a, model) == evaluate(b, model)
    raise TypeError(f"Expresión desconocida: {e!r}")


def symbols_in(e: Expr) -> set[Symbol]:
    match e:
        case Symbol():
            return {e}
        case Not(a):
            return symbols_in(a)
        case And(a, b) | Or(a, b) | Implies(a, b) | Iff(a, b):
            return symbols_in(a) | symbols_in(b)
    raise TypeError(f"Expresión desconocida: {e!r}")


def tt_entails(kb: Expr, query: Expr) -> bool:
    """KB ⊨ α por enumeración de modelos: α es verdadera en todo modelo donde KB lo es.

    Correcto y completo, pero O(2ⁿ) en el número de símbolos: viable hasta ~20 símbolos.
    """
    syms = sorted(symbols_in(kb) | symbols_in(query), key=lambda s: s.name)
    for values in itertools.product([False, True], repeat=len(syms)):
        model = dict(zip(syms, values, strict=True))
        if evaluate(kb, model) and not evaluate(query, model):
            return False
    return True


def is_satisfiable(e: Expr) -> bool:
    syms = sorted(symbols_in(e), key=lambda s: s.name)
    return any(
        evaluate(e, dict(zip(syms, values, strict=True)))
        for values in itertools.product([False, True], repeat=len(syms))
    )


@dataclass(frozen=True)
class HornClause:
    """Cláusula definida: premisa₁ ∧ … ∧ premisaₙ ⇒ conclusión (un hecho si no hay premisas)."""

    premises: tuple[str, ...]
    conclusion: str


def forward_chaining(clauses: Iterable[HornClause], query: str) -> tuple[bool, list[str]]:
    """PL-FC-Entails: dispara reglas cuyas premisas ya se conocen hasta derivar `query` o agotar.

    Tiempo lineal en el tamaño de la base (cada premisa se descuenta una vez). Solo vale para
    cláusulas de Horn, a cambio de esa eficiencia. Devuelve (¿se deriva?, orden de inferencia).
    """
    clauses = list(clauses)
    count = [len(c.premises) for c in clauses]
    uses: dict[str, list[int]] = {}
    for i, c in enumerate(clauses):
        for p in c.premises:
            uses.setdefault(p, []).append(i)
    agenda = deque(c.conclusion for c in clauses if not c.premises)
    inferred: set[str] = set()
    order: list[str] = []
    while agenda:
        p = agenda.popleft()
        if p in inferred:
            continue
        inferred.add(p)
        order.append(p)
        if p == query:
            return True, order
        for i in uses.get(p, []):
            count[i] -= 1
            if count[i] == 0:
                agenda.append(clauses[i].conclusion)
    return False, order
