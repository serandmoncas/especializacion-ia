"""Ruta de conceptos: candidatos desde la página del tema y orden por prerrequisitos."""

import numpy as np

from research_paths.retrieval import Encoder, cosine_scores

MUTUAL_BONUS = 0.1  # un concepto que enlaza de vuelta al tema sube un poco


def concept_text(title: str, extract: str) -> str:
    name = title.replace("_", " ")
    return f"{name}. {extract}" if extract else name


def resolve_topic(query: str, wiki) -> tuple[str | None, list[str]]:
    """Título canónico del tema, o None y hasta 5 sugerencias (no existe o es desambiguación)."""
    title = wiki.resolve([query])[query]
    return (title, []) if title else (None, wiki.search(query, 5))


def candidate_concepts(topic: str, wiki, encoder: Encoder, k: int = 8, pool: int = 30) -> list[str]:
    """Conceptos enlazados desde el tema, por similitud con él (+ bono si enlazan de vuelta)."""
    links = [x for x in wiki.links(topic) if x != topic]
    if not links:
        return []
    texts = wiki.extracts([topic, *links])
    have = [x for x in links if texts.get(x)]
    if not have:
        return []
    query = encoder([concept_text(topic, texts[topic])], "query")[0]
    sims = cosine_scores(query, encoder([concept_text(x, texts[x]) for x in have], "passage"))
    scored = []
    for i in np.argsort(-sims, kind="mergesort")[:pool]:
        mutual = topic in wiki.links(have[i])
        scored.append((float(sims[i]) + MUTUAL_BONUS * mutual, have[i]))
    scored.sort(key=lambda s: -s[0])
    return [name for _, name in scored[:k]]


def prereq_matrix(
    concepts: list[str],
    texts: dict[str, str],
    model,
    tokenizer,
    *,
    max_length: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """P[i, j] = P(concepts[j] es prerrequisito de concepts[i]); diagonal 0."""
    from research_paths.textclf import predict_proba

    n = len(concepts)
    P = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if pairs:
        a = [concept_text(concepts[i], texts.get(concepts[i], "")) for i, _ in pairs]
        b = [concept_text(concepts[j], texts.get(concepts[j], "")) for _, j in pairs]
        probs = predict_proba(model, tokenizer, a, b, max_length=max_length, device=device)
        for (i, j), p in zip(pairs, probs, strict=True):
            P[i, j] = p
    return P


def order_route(concepts: list[str], prereq_proba: np.ndarray) -> list[str]:
    """Orden de lectura: primero lo que es prerrequisito de más conceptos.

    `prereq_proba[i, j]` = P(concepts[j] es prerrequisito de concepts[i]).

    TODO(usuario): esta es la versión mínima (contar «victorias» por encima de 0,5). Las
    probabilidades por pares pueden ser inconsistentes (A→B, B→C y C→A). Alternativas: orden
    topológico tras romper cada ciclo por su arista más débil, Borda ponderado por probabilidad,
    o un ranking tipo Bradley-Terry. Debe devolver una permutación de `concepts`.
    """
    P = np.array(prereq_proba, dtype=float)
    np.fill_diagonal(P, 0)
    wins = (P > 0.5).sum(axis=0)
    order = sorted(range(len(concepts)), key=lambda j: (-wins[j], j))
    return [concepts[j] for j in order]


def prerequisite_of(
    concepts: list[str], P: np.ndarray, threshold: float = 0.5
) -> dict[str, list[str]]:
    """Para cada concepto, los conceptos de los que es prerrequisito."""
    return {
        c: [concepts[i] for i in range(len(concepts)) if i != j and P[i, j] > threshold]
        for j, c in enumerate(concepts)
    }
