"""CLI: `uv run python -m research_paths ruta "tema"` → conceptos en orden y lecturas."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from research_paths.concepts import (
    candidate_concepts,
    order_route,
    prereq_matrix,
    prerequisite_of,
    resolve_topic,
)
from research_paths.http import MissingKeyError, require_env
from research_paths.openalex import KEY_URL
from research_paths.readings import start_here_score, works_frame
from research_paths.retrieval import cosine_scores

MODULE = "02-semestre-2/02-pln"
KINDS = {"review": "revisión", "book": "libro", "article": "artículo"}


def find_readings(
    concept_es: str,
    concept_en: str | None,
    oa,
    encoder,
    intro,
    *,
    k: int = 3,
    n: int = 25,
    device: str = "cpu",
) -> pd.DataFrame:
    """Candidatos de OpenAlex (búsqueda en es y en) ordenados por `start_here_score`."""
    from research_paths.textclf import predict_proba

    works = []
    for query in dict.fromkeys(q for q in (concept_es, concept_en) if q):
        works += oa.works(filter="type:article|review|book", search=query, max_results=n)
    cands = works_frame(works).drop_duplicates("id")
    cands = cands[cands.title != ""].reset_index(drop=True)
    if cands.empty:
        return cands
    texts = cands.text.tolist()
    p = predict_proba(*intro, texts, device=device)
    rel = cosine_scores(encoder([concept_es], "query")[0], encoder(texts, "passage"))
    cands = cands.assign(p_intro=p, score=start_here_score(rel, p, cands.citations.to_numpy()))
    return cands.sort_values("score", ascending=False, kind="mergesort").head(k)


def _reason(row) -> str:
    kind = KINDS.get(row.type, row.type or "trabajo")
    p = f"{row.p_intro:.2f}".replace(".", ",")
    return f"{kind}, {row.citations} citas, P(introductorio) {p}"


def _load_models(models_dir: Path):
    from research_paths.textclf import load_classifier

    paths = [models_dir / "prereq-es", models_dir / "intro"]
    if not all(p.is_dir() for p in paths):
        return None
    return tuple(load_classifier(p) for p in paths)


def main(
    argv=None,
    *,
    wiki=None,
    oa=None,
    encoder=None,
    prereq=None,
    intro=None,
    models_dir=None,
    device=None,
    out=None,
) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(
        prog="research_paths", description="Por dónde empezar un tema."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    ruta = sub.add_parser("ruta", help="ruta de conceptos y lecturas para un tema")
    ruta.add_argument(
        "tema", help="título de Wikipedia en español, p. ej. «Aprendizaje por refuerzo»"
    )
    ruta.add_argument("--conceptos", type=int, default=8)
    ruta.add_argument("--lecturas", type=int, default=3)
    ruta.add_argument("--sin-lecturas", action="store_true", help="no consultar OpenAlex")
    args = parser.parse_args(argv)

    from common import data_dir

    base = data_dir(MODULE)
    if prereq is None or intro is None:
        models_dir = Path(models_dir) if models_dir else base / "modelos"
        loaded = _load_models(models_dir)
        if loaded is None:
            print(
                f"No hay modelos entrenados en {models_dir}. Corre primero:\n"
                f"  uv run python {MODULE}/proyecto/run.py",
                file=out,
            )
            return 2
        prereq, intro = loaded
    if device is None:
        from research_paths.textclf import default_device

        device = default_device()
    try:
        if oa is None and not args.sin_lecturas:
            from research_paths.http import CachedClient
            from research_paths.openalex import OpenAlex

            key = require_env("OPENALEX_API_KEY", KEY_URL)
            oa = OpenAlex(CachedClient(base / "cache" / "openalex"), key)
    except MissingKeyError as err:
        print(err, file=out)
        return 2
    if wiki is None:
        from research_paths.http import CachedClient
        from research_paths.wiki import Wiki

        wiki = Wiki(CachedClient(base / "cache" / "wiki-es"), "es")
    if encoder is None:
        from research_paths.retrieval import e5_encoder

        encoder = e5_encoder(device=device)
    return _route(args, wiki, oa, encoder, prereq, intro, device, out)


def _route(args, wiki, oa, encoder, prereq, intro, device, out) -> int:
    title, suggestions = resolve_topic(args.tema, wiki)
    if title is None:
        print(
            f"No encontré «{args.tema}» en Wikipedia en español (o es una desambiguación).",
            file=out,
        )
        if suggestions:
            print("¿Quisiste decir?: " + "; ".join(suggestions), file=out)
        return 1
    concepts = candidate_concepts(title, wiki, encoder, k=args.conceptos)
    if not concepts:
        print(f"«{title}» no enlaza conceptos con texto: no hay ruta.", file=out)
        return 1
    texts = wiki.extracts(concepts)
    P = prereq_matrix(concepts, texts, *prereq, device=device)
    route = order_route(concepts, P)
    before = prerequisite_of(concepts, P)
    english = {} if args.sin_lecturas else wiki.langlinks(route, "en")
    print(f"Ruta para «{title}» ({len(route)} conceptos)\n", file=out)
    for i, concept in enumerate(route, 1):
        print(f"{i}. {concept}", file=out)
        if before[concept]:
            print(f"   Prerrequisito de: {', '.join(before[concept])}", file=out)
        if args.sin_lecturas:
            continue
        found = find_readings(
            concept, english.get(concept), oa, encoder, intro, k=args.lecturas, device=device
        )
        print("   Lecturas:", file=out)
        if found.empty:
            print("   - sin lecturas encontradas", file=out)
        for row in found.itertuples():
            year = row.year or "s. f."
            print(f"   - «{row.title}» ({year}) — {_reason(row)}", file=out)
        print(file=out)
    return 0
