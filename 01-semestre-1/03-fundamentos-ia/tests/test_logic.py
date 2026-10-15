from ai_foundations.logic import (
    HornClause,
    conjoin,
    evaluate,
    forward_chaining,
    iff,
    is_satisfiable,
    symbols,
    tt_entails,
)

P, Q, R = symbols("P Q R")


def test_evaluate_connectives():
    m = {P: True, Q: False}
    assert evaluate(P | Q, m) and not evaluate(P & Q, m)
    assert not evaluate(P >> Q, m) and evaluate(Q >> P, m)
    assert not evaluate(iff(P, Q), m) and evaluate(~Q, m)


def test_modus_ponens_and_fallacies():
    assert tt_entails(conjoin([P >> Q, P]), Q)
    assert not tt_entails(conjoin([P >> Q, Q]), P)  # afirmación del consecuente
    assert tt_entails(conjoin([P >> Q, ~Q]), ~P)  # modus tollens


def test_tautology_and_contradiction():
    assert tt_entails(P, P | ~P)
    assert not is_satisfiable(P & ~P)
    assert tt_entails(P & ~P, R)  # de una contradicción se sigue cualquier cosa


def test_wumpus_pit_inference():
    # Mundo del wumpus reducido: brisa en [1,2] sii hay pozo en [1,1]... vecinos.
    B11, B21, P12, P21, P22, P31 = symbols("B11 B21 P12 P21 P22 P31")
    kb = conjoin(
        [
            iff(B11, P12 | P21),
            iff(B21, P12 | P22 | P31),
            ~B11,
            B21,
        ]
    )
    assert tt_entails(kb, ~P12)  # no hay brisa en [1,1] ⇒ no hay pozo en [1,2]
    assert not tt_entails(kb, P22)  # la brisa en [2,1] no dice cuál vecino tiene pozo
    assert tt_entails(kb, P22 | P31)


def test_forward_chaining_aima_example():
    kb = [
        HornClause(("P",), "Q"),
        HornClause(("L", "M"), "P"),
        HornClause(("B", "L"), "M"),
        HornClause(("A", "P"), "L"),
        HornClause(("A", "B"), "L"),
        HornClause((), "A"),
        HornClause((), "B"),
    ]
    ok, order = forward_chaining(kb, "Q")
    assert ok and order[-1] == "Q"
    assert set(order) == {"A", "B", "L", "M", "P", "Q"}
    assert not forward_chaining(kb, "Z")[0]


def test_forward_chaining_agrees_with_model_checking():
    kb = [HornClause((), "A"), HornClause(("A",), "B"), HornClause(("B", "C"), "D")]
    A, B, C, D = symbols("A B C D")
    expr = conjoin([A, A >> B, (B & C) >> D])
    for name, sym in [("B", B), ("D", D)]:
        assert forward_chaining(kb, name)[0] == tt_entails(expr, sym)
