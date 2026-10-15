"""CocoaMoniliaDataSet: anotaciones COCO → tablas de fotos y de mazorcas, sin duplicados."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from applied_cv.boxes import box_iou

CLASSES = ("h0", "m1", "m2", "m3")  # sana, jorobas, manchas café, esporulación
_IMG = re.compile(r"^IMG_(\d{8})_(\d{6})")
_DATE_TIME = re.compile(r"^(\d{8})_(\d{6})")
_EPOCH = re.compile(r"^(\d{13})")
_SEQUENCE = re.compile(r"^IMG_E?(\d+)\.")


def parse_filename(name: str) -> tuple[str, datetime | None, int | None]:
    """(formato de celular, fecha-hora de captura, número de secuencia) a partir del nombre.

    El formato del nombre delata el celular: en el dataset casi todas las h0 de un formato y casi
    todas las m1 de otro, por eso se guarda como metadato para controlar el atajo.
    """
    if m := _IMG.match(name):
        return "IMG", datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S"), None
    if m := _DATE_TIME.match(name):
        return "fecha_hora", datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S"), None
    if m := _EPOCH.match(name):
        moment = datetime.fromtimestamp(int(m[1]) / 1000, UTC).replace(tzinfo=None)
        return "epoch", moment, None
    if m := _SEQUENCE.match(name):
        return "secuencia", None, int(m[1])
    return "otro", None, None


def _dedupe_pods(pods: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    keep = []
    for _, group in pods.groupby("image", sort=False):
        boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(float)
        kept: list[int] = []
        for i in range(len(group)):
            if not kept or box_iou(boxes[i], boxes[kept]).max() <= iou_threshold:
                kept.append(i)
        keep.extend(group.index[kept])
    return pods.loc[sorted(keep)].reset_index(drop=True)


def load_annotations(
    root: str | Path, dedupe_iou: float = 0.9
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee los cuatro `instances_<clase>.json` y devuelve (fotos, mazorcas).

    Los ids de imagen se reinician en cada archivo y algunas fotos están en dos carpetas: la foto se
    identifica por su nombre y se queda con la primera carpeta. Las mazorcas repetidas (IoU >
    `dedupe_iou` con otra de la misma foto) se descartan.
    Cajas COCO [x, y, w, h] → [x1, y1, x2, y2].
    """
    root = Path(root)
    images: dict[str, dict] = {}
    rows = []
    for folder in CLASSES:
        coco = json.loads((root / "COCO_annotations" / f"instances_{folder}.json").read_text())
        label_of = {c["id"]: c["name"] for c in coco["categories"]}
        name_of = {}
        for im in coco["images"]:
            name = im["file_name"]
            name_of[im["id"]] = name
            if name not in images:
                camera, timestamp, seq = parse_filename(name)
                images[name] = {
                    "image": name,
                    "folder": folder,
                    "path": f"cocoapods_images/{folder}/{name}",
                    "width": im["width"],
                    "height": im["height"],
                    "camera": camera,
                    "timestamp": timestamp,
                    "seq": seq,
                }
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            rows.append((name_of[a["image_id"]], x, y, x + w, y + h, label_of[a["category_id"]]))
    pods = pd.DataFrame(rows, columns=["image", "x1", "y1", "x2", "y2", "label"])
    pods = _dedupe_pods(pods, dedupe_iou)
    pods.insert(0, "pod_id", np.arange(len(pods)))
    return pd.DataFrame(list(images.values())), pods
