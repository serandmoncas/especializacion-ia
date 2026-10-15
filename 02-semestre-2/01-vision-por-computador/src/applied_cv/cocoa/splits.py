"""Particiones honestas: fotos de la misma ráfaga (misma mazorca o árbol) nunca en dos partes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Intervalos entre fotos consecutivas del mismo celular en el dataset: mediana 4 s, 90 % ≤ 72 s,
# 95 % ≤ 134 s. Con 120 s una ráfaga es «la misma sesión frente a la misma mazorca o árbol».
GAP_SECONDS = 120
# Las 232 fotos con nombre epoch (ms) caen en 2 s, con 8 ms de mediana entre ellas: es la hora de
# exportación, no la de captura. Un celular cuya mediana entre fotos es < 1 s no tiene hora útil.
MIN_MEDIAN_GAP = 1.0


def assign_bursts(
    images: pd.DataFrame,
    gap_seconds: float = GAP_SECONDS,
    seq_gap: int = 3,
    min_median_gap: float = MIN_MEDIAN_GAP,
) -> pd.Series:
    """Id de ráfaga por foto.

    Con hora: fotos del mismo celular separadas por ≤ `gap_seconds` de la anterior. Sin hora pero
    numeradas (IMG_E####): números a ≤ `seq_gap` del anterior. Sin ninguna de las dos:
    cada foto sola. La hora de un celular cuya mediana entre fotos consecutivas es menor que
    `min_median_gap` segundos no es de captura y se ignora.
    """
    burst = pd.Series(-1, index=images.index, dtype=int)
    next_id = 0
    timed = images[images["timestamp"].notna()].sort_values(["camera", "timestamp"])
    median_gap = timed.groupby("camera")["timestamp"].apply(
        lambda t: t.diff().dt.total_seconds().median()
    )
    export_only = set(median_gap[median_gap < min_median_gap].index)
    timed = timed[~timed["camera"].isin(export_only)]
    prev_camera, prev_time = None, None
    for idx, row in timed.iterrows():
        if row.camera != prev_camera or (row.timestamp - prev_time).total_seconds() > gap_seconds:
            next_id += 1
        burst[idx] = next_id
        prev_camera, prev_time = row.camera, row.timestamp
    numbered = images[images["timestamp"].isna() & images["seq"].notna()].sort_values("seq")
    numbered = numbered[~numbered["camera"].isin(export_only)]
    prev_seq = None
    for idx, row in numbered.iterrows():
        if prev_seq is None or row.seq - prev_seq > seq_gap:
            next_id += 1
        burst[idx] = next_id
        prev_seq = row.seq
    for idx in images.index[burst == -1]:
        next_id += 1
        burst[idx] = next_id
    return burst


def burst_split(images: pd.DataFrame, bursts: pd.Series, seed: int = 42) -> pd.Series:
    """≈ 70/10/20 agrupada por ráfaga y estratificada por la carpeta (clase de la foto)."""
    y, groups = images["folder"].to_numpy(), np.asarray(bursts)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    train, val = next(inner.split(np.zeros(len(rest)), y[rest], groups[rest]))
    part = np.empty(len(y), dtype=object)
    part[test], part[rest[val]], part[rest[train]] = "test", "val", "train"
    return pd.Series(part, index=images.index)


def read_split_lists(directory: str | Path) -> dict[str, list[str]]:
    """Nombres de archivo de `train.txt`, `val.txt` y `test.txt` publicados con el dataset."""
    directory = Path(directory)
    # Una ruta por línea: hay nombres con espacios («m0 (95).jpg»), así que no se parte por blancos.
    return {
        part: [
            Path(line.strip()).name
            for line in (directory / f"{part}.txt").read_text().splitlines()
            if line.strip()
        ]
        for part in ("train", "val", "test")
    }


def published_split(
    images: pd.DataFrame, lists: dict[str, list[str]]
) -> tuple[pd.Series, list[str]]:
    """Parte de cada foto según las listas del autor.

    Una foto que está en dos carpetas puede aparecer en dos partes (p. ej. train y test): eso ya es
    una fuga en la partición publicada, así que esas fotos quedan `fuera` y se informan.
    """
    parts_of: dict[str, set[str]] = {}
    for part, names in lists.items():
        for name in names:
            parts_of.setdefault(name, set()).add(part)
    conflicts = sorted(n for n in images["image"] if len(parts_of.get(n, ())) > 1)
    part = images["image"].map(
        lambda n: next(iter(parts_of[n])) if len(parts_of.get(n, ())) == 1 else "fuera"
    )
    return part, conflicts
