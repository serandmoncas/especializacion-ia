"""RefD (Liang et al., 2015): ¿A «se refiere» a B más de lo que B se refiere a A?

RefD(A, B) = Σ_c w(c, A)·r(c, B) / Σ_c w(c, A) − Σ_c w(c, B)·r(c, A) / Σ_c w(c, B), con
w(c, X) = 1 si X enlaza a c y r(c, Y) = 1 si c enlaza a Y o c = Y. Los c se limitan a `concepts`
(solo se descargan los enlaces de esos conceptos, no de toda Wikipedia).
"""

import numpy as np
import pandas as pd


def _refers(x: str, y: str, links: dict[str, set[str]], concepts: set[str]) -> float:
    outgoing = links.get(x, set()) & concepts
    if not outgoing:
        return 0.0
    hits = sum(1 for c in outgoing if c == y or y in links.get(c, set()))
    return hits / len(outgoing)


def refd(a: str, b: str, links: dict[str, set[str]], concepts: set[str]) -> float:
    """> 0: b se parece más a un prerrequisito de a que al revés."""
    return _refers(a, b, links, concepts) - _refers(b, a, links, concepts)


def refd_scores(
    pairs: pd.DataFrame,
    links: dict[str, set[str]],
    concepts: set[str],
    col_a: str = "a_es",
    col_b: str = "b_es",
) -> np.ndarray:
    return np.array(
        [refd(a, b, links, concepts) for a, b in zip(pairs[col_a], pairs[col_b], strict=True)]
    )
