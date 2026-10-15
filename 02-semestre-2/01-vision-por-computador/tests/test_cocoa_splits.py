from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from applied_cv.cocoa.splits import assign_bursts, burst_split, published_split, read_split_lists


def _images(rows):
    return pd.DataFrame(rows, columns=["image", "folder", "camera", "timestamp", "seq"])


def test_assign_bursts_gap_camera_sequence_and_singletons():
    t0 = datetime(2024, 2, 26, 14, 0, 0)
    images = _images(
        [
            ("a", "h0", "IMG", t0, None),
            ("b", "h0", "IMG", t0 + timedelta(seconds=90), None),  # misma ráfaga
            ("c", "h0", "IMG", t0 + timedelta(seconds=300), None),  # nueva: 210 s > 120 s
            ("d", "m1", "epoch", t0 + timedelta(seconds=95), None),  # otro celular: nueva
            ("e", "m2", "secuencia", None, 2361),
            ("f", "m2", "secuencia", None, 2363),  # salto 2 ≤ 3: misma ráfaga
            ("g", "m2", "secuencia", None, 2385),  # salto 22: nueva
            ("h", "m1", "otro", None, None),  # sin hora ni número: sola
            ("i", "m1", "otro", None, None),
        ]
    )
    b = assign_bursts(images, gap_seconds=120, seq_gap=3)
    assert b[0] == b[1]
    assert len({b[0], b[2], b[3]}) == 3
    assert b[4] == b[5] != b[6]
    assert b[7] != b[8] and b[7] not in set(b[:7])
    assert b.nunique() == 7


def test_burst_split_keeps_bursts_together_and_fractions():
    rng = np.random.default_rng(0)
    folders = rng.choice(["h0", "m1", "m2", "m3"], 100)
    images = pd.DataFrame({"image": [f"x{i}" for i in range(200)], "folder": np.repeat(folders, 2)})
    bursts = pd.Series(np.repeat(np.arange(100), 2))
    part = burst_split(images, bursts, seed=42)
    assert set(part) == {"train", "val", "test"}
    assert part.groupby(bursts).nunique().max() == 1
    share = part.value_counts(normalize=True)
    assert 0.6 < share["train"] < 0.8 and 0.05 < share["val"] < 0.16 and 0.14 < share["test"] < 0.26


def test_published_split_conflicts(cocoa_raw):
    lists = read_split_lists(cocoa_raw)
    assert sum(len(v) for v in lists.values()) == 12
    images = pd.DataFrame(
        {"image": ["IMG_20240227_090000000.jpg", "1728667861304.jpg", "nuevo.jpg"]}
    )
    part, conflicts = published_split(images, lists)
    assert conflicts == ["IMG_20240227_090000000.jpg"]
    assert part.tolist() == ["fuera", "train", "fuera"]


def test_assign_bursts_ignores_export_timestamps():
    # 5 fotos «epoch» con 10 ms entre sí: es la hora de exportación, no de captura → cada una sola.
    t0 = datetime(2024, 10, 11, 17, 31, 1)
    rows = [(f"e{i}", "h0", "epoch", t0 + timedelta(milliseconds=10 * i), None) for i in range(5)]
    rows += [
        (name, "h0", "IMG", t0 + timedelta(seconds=sec), None)
        for name, sec in (("a", 0), ("b", 0.5), ("c", 30), ("d", 60))
    ]
    b = assign_bursts(_images(rows))
    assert b[:5].nunique() == 5
    assert (
        b[5] == b[6]
    )  # en IMG un par a 0,5 s (HDR) sí es ráfaga: su mediana de intervalos es normal
