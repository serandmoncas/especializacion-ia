"""Preparación única: fotos reducidas para el detector y recortes cuadrados por mazorca."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from applied_cv.cocoa.data import load_annotations

EXIF_ORIENTATION = 0x0112


def square_crop_box(
    box: tuple[float, float, float, float], width: int, height: int, margin: float = 0.1
) -> tuple[int, int, int, int]:
    """Cuadrado centrado en la caja, de lado max(ancho, alto)·(1 + 2·margen), recortado a la imagen.

    Cuadrado para no deformar la mazorca (alta y angosta) al llevarla a 224×224.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    left, top = cx - side / 2, cy - side / 2
    return (
        max(0, round(left)),
        max(0, round(top)),
        min(width, round(left + side)),
        min(height, round(top + side)),
    )


def crop_pod(
    image: Image.Image, box: tuple[float, float, float, float], size: int, margin: float = 0.1
) -> Image.Image:
    """Recorte cuadrado de una mazorca llevado a `size`×`size` con remuestreo bilineal.

    Es la única forma de recortar del proyecto (preparación, sistema completo y demo): un
    clasificador que en inferencia ve recortes remuestreados de otra forma mide otra cosa.
    """
    square = square_crop_box(box, image.width, image.height, margin)
    return image.crop(square).resize((size, size), Image.BILINEAR)


def prepare(
    raw_root: str | Path,
    out_dir: str | Path,
    max_side: int = 1024,
    crop_size: int = 224,
    margin: float = 0.1,
    apply_exif: bool = False,
) -> dict:
    """Reduce cada foto a lado máximo `max_side`, recorta cada mazorca y escribe las tablas.

    Las fotos del JSON sin archivo se excluyen y se informan (`missing`); también las que tienen un
    tamaño distinto al del JSON (`size_mismatch`) y el conteo de orientaciones EXIF, porque si las
    anotaciones se hicieron sobre la foto rotada las cajas quedarían desalineadas.
    """
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    images, pods = load_annotations(raw_root)
    missing, mismatch, orientations = [], [], Counter()
    kept_images, kept_pods = [], []
    for row in images.itertuples(index=False):
        path = raw_root / row.path
        if not path.exists():
            missing.append(row.image)
            continue
        with Image.open(path) as opened:
            orientation = opened.getexif().get(EXIF_ORIENTATION, 1)
            im = ImageOps.exif_transpose(opened) if apply_exif else opened.copy()
        orientations[orientation] += 1
        im = im.convert("RGB")
        if im.size != (row.width, row.height):
            mismatch.append(row.image)
        scale = min(1.0, max_side / max(im.size))
        resized = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
        resized.save(out_dir / "images" / row.image, quality=90)
        kept_images.append({**row._asdict(), "scale": scale, "exif": orientation})
        for pod in pods[pods.image == row.image].itertuples(index=False):
            box = (pod.x1 * scale, pod.y1 * scale, pod.x2 * scale, pod.y2 * scale)
            crop_path = f"crops/{pod.pod_id}.jpg"
            crop_pod(resized, box, crop_size, margin).save(out_dir / crop_path, quality=95)
            kept_pods.append(
                {
                    "pod_id": pod.pod_id,
                    "image": pod.image,
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "label": pod.label,
                    "crop": crop_path,
                }
            )
    pd.DataFrame(kept_images).to_csv(out_dir / "images.csv", index=False)
    pd.DataFrame(kept_pods).to_csv(out_dir / "pods.csv", index=False)
    summary = {
        "images": len(kept_images),
        "pods": len(kept_pods),
        "missing": missing,
        "size_mismatch": mismatch,
        "exif_orientations": dict(orientations),
        "same_name_other_photo": pods.attrs["same_name_other_photo"],
        "label_conflicts": pods.attrs["label_conflicts"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
