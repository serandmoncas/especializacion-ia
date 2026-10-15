"""CLI: temas de un texto, ficha de un libro y recomendaciones explicadas."""

import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from book_topics import MODULE
from book_topics.artifacts import load_bundle, predict_topics, save_bundle, train_bundle
from book_topics.recommend import explain, profile_from_titles, topic_scores_with_tiebreak

_PUNCT = re.compile(r"[^\w\s]")


def normalize(title: str) -> str:
    return " ".join(_PUNCT.sub(" ", title.lower()).split())


def find_book(query: str, titles: pd.Series) -> tuple[int | None, list[str]]:
    """Exacto o prefijo → parcial → difflib. Devuelve la mejor posición y hasta 3 alternativas."""
    norm = titles.map(normalize).tolist()
    q = normalize(query)
    exact = [i for i, t in enumerate(norm) if t == q or t.startswith(q + " ")]
    if len(exact) == 1:
        return exact[0], []
    contains = exact or [i for i, t in enumerate(norm) if q and q in t]
    if contains:
        return contains[0], [titles.iloc[i] for i in contains[1:4]]
    close = difflib.get_close_matches(q, norm, n=4, cutoff=0.6)
    if close:
        return norm.index(close[0]), [titles.iloc[norm.index(c)] for c in close[1:]]
    return None, [
        titles.iloc[norm.index(a)] for a in difflib.get_close_matches(q, norm, n=3, cutoff=0.0)
    ]


def _paths(bundle_path: Path | None, data_dir: Path | None) -> tuple[Path, Path]:
    if bundle_path is not None and data_dir is not None:
        return bundle_path, data_dir
    from common import data_dir as module_data_dir

    d = data_dir or module_data_dir(MODULE)
    return bundle_path or d / "modelos" / "bundle.joblib", d


def main(
    argv: list[str] | None = None, *, bundle_path: Path | None = None, data_dir: Path | None = None
) -> int:
    parser = argparse.ArgumentParser(
        prog="book_topics", description="Temas de libros y recomendaciones"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("entrenar", help="entrena y guarda los modelos")
    sub.add_parser("temas", help="temas de un título o una descripción").add_argument("texto")
    sub.add_parser("libro", help="ficha de un libro del catálogo").add_argument("titulo")
    rec = sub.add_parser("recomendar", help="recomendaciones a partir de libros leídos")
    rec.add_argument("--lei", nargs="+", required=True)
    rec.add_argument("--no-me-gusto", nargs="*", default=[])
    rec.add_argument("-n", type=int, default=10)
    args = parser.parse_args(argv)
    bundle_path, data_dir = _paths(bundle_path, data_dir)

    if args.cmd == "entrenar":
        save_bundle(train_bundle(data_dir, data_dir.parent / "temas.yaml"), bundle_path)
        print(f"Modelos guardados en {bundle_path}")
        return 0
    try:
        b = load_bundle(bundle_path)
    except FileNotFoundError as e:
        print(e)
        return 1
    names = [t.nombre for t in b.topics]
    titles = b.books.title

    if args.cmd == "temas":
        text = args.texto
        pos, _ = find_book(args.texto, titles)
        if pos is not None and normalize(titles.iloc[pos]).startswith(normalize(args.texto)):
            text = f"{titles.iloc[pos]}. {b.descriptions.get(b.books.book_id.iloc[pos], '')}"
            print(f"Usando la descripción de «{titles.iloc[pos]}».")
        P, assigned = predict_topics(b, [text])
        for t in np.argsort(-P[0])[:6]:
            print(f"{'✓' if assigned[0, t] else ' '} {names[t]:<28} {P[0, t]:.2f}")
        if b.model.low_confidence([text])[0]:
            print(
                "Aviso: baja confianza. El texto tiene pocas palabras conocidas o no parece estar "
                "en inglés (el modelo se entrenó con descripciones en inglés): tomar con cautela."
            )
        return 0

    if args.cmd == "libro":
        pos, alts = find_book(args.titulo, titles)
        if pos is None:
            print(f"No encontré «{args.titulo}». ¿Quisiste decir: {'; '.join(alts)}?")
            return 1
        print(f"{titles.iloc[pos]} — {b.books.authors.iloc[pos]}")
        if alts:
            print(f"También coincide con: {'; '.join(alts)}")
        real = [names[t] for t in np.flatnonzero(b.T[pos])]
        print(
            "Temas reales: "
            + (", ".join(real) if real else "sin temas asignados (estantes insuficientes)")
        )
        _, assigned = predict_topics(
            b, [f"{titles.iloc[pos]}. {b.descriptions.get(b.books.book_id.iloc[pos], '')}"]
        )
        print("Temas predichos: " + ", ".join(names[t] for t in np.flatnonzero(assigned[0])))
        vec = b.T[pos] if b.T[pos].any() else assigned[0].astype(float)
        scores = topic_scores_with_tiebreak(vec[None, :], b.T, b.popularity)[0]
        scores[pos] = -np.inf
        print("Parecidos por temas:")
        for i, j in enumerate(np.argsort(-scores)[:10], 1):
            print(f"  {i}. {titles.iloc[j]}")
        return 0

    liked, disliked = [], []
    for queries, out in ((args.lei, liked), (args.no_me_gusto, disliked)):
        for q in queries:
            pos, alts = find_book(q, titles)
            if pos is None:
                print(f"No encontré «{q}» (se ignora). Parecidos: {'; '.join(alts)}")
                continue
            out.append(pos)
    if not liked and not disliked:
        print("Ningún título reconocido.")
        return 1
    profile = profile_from_titles(liked, disliked, b.T)
    if not np.any(profile > 0):
        print("No hay temas suficientes en lo que indicaste: recomiendo por popularidad.")
        scores = b.popularity.astype(float).copy()
    else:
        scores = topic_scores_with_tiebreak(profile[None, :], b.T, b.popularity)[0]
    scores[list(set(liked) | set(disliked))] = -np.inf
    for i, j in enumerate(np.argsort(-scores)[: args.n], 1):
        why = explain(profile, b.T[j], names)
        print(f"{i}. {titles.iloc[j]}" + (f" — porque te gustan: {', '.join(why)}" if why else ""))
    return 0
