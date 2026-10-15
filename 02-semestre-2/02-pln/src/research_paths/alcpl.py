"""AL-CPL (Liang et al., 2018): prerrequisitos entre conceptos, mapeo a Wikipedia ES y particiones.

En cada par (A, B), B es prerrequisito de A. Los autores ampliaron el conjunto con el par invertido
como negativo y con el cierre transitivo, así que una partición aleatoria de pares filtra la
respuesta: se parte por dominio y se quitan los pares que tocan conceptos del otro lado.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DOMAINS = ("data_mining", "geometry", "physics", "precalculus")


def read_pairs(path: Path) -> list[tuple[str, str]]:
    with Path(path).open(newline="") as f:
        return [(r[0].strip(), r[1].strip()) for r in csv.reader(f) if len(r) >= 2]


def load_alcpl(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        pairs = read_pairs(Path(raw_dir) / f"{domain}.pairs")
        positives = set(read_pairs(Path(raw_dir) / f"{domain}.preqs"))
        if missing := positives - set(pairs):
            raise ValueError(f"{domain}: {len(missing)} positivos no están en .pairs")
        rows += [(domain, a, b, int((a, b) in positives)) for a, b in pairs]
    df = pd.DataFrame(rows, columns=["domain", "a", "b", "label"])
    return df.drop_duplicates(["domain", "a", "b"]).reset_index(drop=True)


def map_concepts(df: pd.DataFrame, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, dict]:
    """Agrega `a_es`, `b_es`; descarta pares sin página ES o con ambos conceptos en la misma."""
    out = df.assign(a_es=df.a.map(mapping), b_es=df.b.map(mapping))
    unmapped = out.a_es.isna() | out.b_es.isna()
    collision = ~unmapped & (out.a_es == out.b_es)
    keep = ~unmapped & ~collision
    concepts = set(df.a) | set(df.b)
    stats = {
        "concepts": len(concepts),
        "concepts_unmapped": sum(mapping.get(c) is None for c in concepts),
        "pairs": len(df),
        "pairs_unmapped": int(unmapped.sum()),
        "pairs_collision": int(collision.sum()),
        "pairs_kept": int(keep.sum()),
        "positives_kept": int(out.label[keep].sum()),
    }
    return out[keep].reset_index(drop=True), stats


@dataclass(frozen=True)
class Fold:
    test: str
    val: str
    train: tuple[str, ...]


def folds() -> list[Fold]:
    """Test = dominio i, validación = dominio i+1, entrenamiento = los otros dos."""
    out = []
    for i, test in enumerate(DOMAINS):
        val = DOMAINS[(i + 1) % len(DOMAINS)]
        out.append(Fold(test, val, tuple(d for d in DOMAINS if d not in (test, val))))
    return out


def _concepts(df: pd.DataFrame) -> set[str]:
    return set(df.a_es) | set(df.b_es)


def _without(df: pd.DataFrame, concepts: set[str]) -> pd.DataFrame:
    return df[~df.a_es.isin(concepts) & ~df.b_es.isin(concepts)]


def split_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = df[df.domain == fold.test]
    test_c = _concepts(test)
    val = _without(df[df.domain == fold.val], test_c)
    train = _without(df[df.domain.isin(fold.train)], test_c | _concepts(val))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
