"""Dataset sintético con la estructura de CocoaMoniliaDataSet, para tests y el smoke de run.py."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

LABELS = ("h0", "m1", "m2", "m3", "m4")
CATEGORIES = [{"id": i + 1, "name": n} for i, n in enumerate(LABELS)]
COLORS = {
    "h0": (90, 160, 60),
    "m1": (120, 150, 60),
    "m2": (110, 70, 30),
    "m3": (225, 225, 215),
    "m4": (60, 40, 20),
}
SIZE_MISMATCH = {"m0 (95).jpg"}  # el JSON dice (alto, ancho) intercambiados


def _draw(path: Path, size, pods, rng) -> None:
    w, h = size
    arr = np.full((h, w, 3), (30, 70, 30), np.uint8) + rng.integers(
        0, 20, (h, w, 3), dtype=np.uint8
    )
    for (x1, y1, x2, y2), label in pods:
        arr[y1:y2, x1:x2] = COLORS[label]
        if label == "m1":  # «jorobas»: puntos claros
            arr[y1 + 2 : y2 - 2 : 6, x1 + 2 : x2 - 2 : 6] = (200, 210, 120)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=95)


def make_cocoa_dataset(root: Path, n_extra: int = 0, seed: int = 0) -> Path:
    """Crea `root/CocoaMoniliaDataSet/` y `root/{train,val,test}.txt`.

    Casos: ids que se reinician por archivo; una foto en dos carpetas (y en train y test de las
    listas); una mazorca duplicada; una m4; formatos epoch, fecha_hora, IMG, IMG_E y otro; una
    imagen del JSON sin archivo; una mazorca pegada al borde; un tamaño intercambiado en el JSON.
    `n_extra` agrega por clase fotos con una mazorca, cada una en su propia ráfaga.
    """
    rng = np.random.default_rng(seed)
    base = root / "CocoaMoniliaDataSet"
    spec = {
        "h0": [
            ("1728667861304.jpg", (160, 120), [((40, 20, 30, 80), "h0")], True),
            ("IMG_20240226_143016382_HDR.jpg", (120, 160), [((0, 30, 40, 60), "h0")], True),
            (
                "IMG_20240226_143020000.jpg",
                (160, 120),
                [((60, 20, 30, 80), "h0"), ((100, 10, 30, 90), "m4")],
                True,
            ),
            ("IMG_20240227_090000000.jpg", (160, 120), [((20, 20, 30, 80), "h0")], True),
        ],
        "m1": [
            ("20241011_110458.jpg", (160, 120), [((50, 15, 30, 85), "m1")], True),
            ("IMG_20240227_090000000.jpg", (160, 120), [((20, 20, 30, 80), "h0")], True),
            ("IMG_E2361.jpg", (160, 120), [((40, 20, 30, 80), "m1")], True),
            ("IMG_E2362.jpg", (160, 120), [((45, 20, 30, 80), "m1")], True),
            ("m0 (95).jpg", (160, 120), [((70, 20, 30, 80), "m1")], True),
        ],
        "m2": [
            (
                "IMG_20240229_100000000.jpg",
                (160, 120),
                [((30, 20, 30, 80), "m2"), ((30, 20, 30, 80), "m2")],
                True,
            ),
        ],
        "m3": [
            ("IMG_20240229_100500000.jpg", (160, 120), [((90, 20, 30, 80), "m3")], True),
            ("IMG_20240229_110000000.jpg", (160, 120), [((10, 10, 30, 80), "m3")], False),
        ],
    }
    for k in range(n_extra):
        for j, label in enumerate(("h0", "m1", "m2", "m3")):
            minutes = 10 * (4 * k + j)  # 10 min entre fotos: cada una es una ráfaga
            name = f"IMG_20240301_{8 + minutes // 60:02d}{minutes % 60:02d}00{k:03d}.jpg"
            if label == "h0" and k % 2:  # la mitad de las sanas con el «otro celular» (epoch)
                name = f"{1709280000000 + 60_000 * minutes}.jpg"
            x = int(rng.integers(10, 110))
            spec[label].append((name, (160, 120), [((x, 15, 35, 90), label)], True))
    names = []
    for folder, items in spec.items():
        images, annotations = [], []
        for i, (name, (w, h), pods, exists) in enumerate(items, start=1):
            json_w, json_h = (h, w) if name in SIZE_MISMATCH else (w, h)
            images.append({"id": i, "width": json_w, "height": json_h, "file_name": name})
            for (x, y, bw, bh), label in pods:
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": i,
                        "category_id": LABELS.index(label) + 1,
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "segmentation": [],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                    }
                )
            if exists:
                boxes = [((x, y, x + bw, y + bh), label) for (x, y, bw, bh), label in pods]
                _draw(base / "cocoapods_images" / folder / name, (w, h), boxes, rng)
            names.append(f"CocoaMoniliaDataSet/cocoapods_images/{folder}/{name}")
        coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
        out = base / "COCO_annotations" / f"instances_{folder}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(coco))
    parts = {"train": [], "val": [], "test": []}
    for i, n in enumerate(names):
        if n.endswith("m1/IMG_20240227_090000000.jpg"):
            parts["test"].append(n)  # la copia de m1 en test; la de h0 cae en val: conflicto
        else:
            parts[("train", "train", "train", "val", "test")[i % 5]].append(n)
    for part, items in parts.items():
        (root / f"{part}.txt").write_text("\n".join(items) + "\n")
    return root


@pytest.fixture
def cocoa_raw(tmp_path) -> Path:
    return make_cocoa_dataset(tmp_path / "raw")


@pytest.fixture
def cocoa_raw_large(tmp_path) -> Path:
    """Con 16 fotos extra por clase: suficiente para entrenar y partir en el smoke de run.py."""
    return make_cocoa_dataset(tmp_path / "raw", n_extra=16)
