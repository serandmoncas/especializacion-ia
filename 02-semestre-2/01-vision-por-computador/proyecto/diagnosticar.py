"""Demo: foto con una o varias mazorcas → cajas, etapa estimada y Grad-CAM de cada mazorca.

Uso:
    uv run python 02-semestre-2/01-vision-por-computador/proyecto/diagnosticar.py foto.jpg
Requiere haber ejecutado proyecto/run.py (guarda los modelos en data/modelos/). Los modelos se
cargan con torch.load(weights_only=True): solo pesos, nunca código.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from applied_cv import MODULE
from applied_cv.cocoa.classifiers import CropDataset, build_resnet, eval_transform, predict_proba
from applied_cv.cocoa.data import CLASSES
from applied_cv.cocoa.detector import build_detector
from applied_cv.cocoa.prepare import crop_pod
from applied_cv.gradcam import grad_cam
from applied_dl.training import get_device
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.transforms.functional import to_tensor

from common import data_dir

OUT = Path(__file__).resolve().parent / "resultados"
STAGE_NAMES = {"h0": "sana", "m1": "jorobas (temprana)", "m2": "manchas café", "m3": "esporulación"}


def load_models(models_dir: Path):
    config_path = models_dir / "modelos.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text())
    detector = build_detector(False, config["det_min_size"], config["det_max_size"])
    detector.load_state_dict(torch.load(models_dir / "detector.pt", weights_only=True))
    classifier = build_resnet(len(CLASSES), pretrained=False)
    classifier.load_state_dict(torch.load(models_dir / "clasificador.pt", weights_only=True))
    return detector.eval(), classifier.eval(), config["umbral"]


def main(argv: list[str] | None = None, *, models_dir: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("foto")
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--min-score", type=float, default=0.5, help="puntaje mínimo del detector")
    args = parser.parse_args(argv)
    path = Path(args.foto)
    if not path.exists():
        print(f"No existe el archivo {path}.")
        return 2
    if not path.is_file():
        print(f"{path} no es un archivo.")
        return 2
    try:
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError):  # OSError: archivo truncado o ilegible
        print(f"{path} no es una imagen que se pueda abrir.")
        return 2
    try:
        detector, classifier, threshold = load_models(models_dir or data_dir(MODULE) / "modelos")
    except FileNotFoundError:
        print("Faltan los modelos: ejecuta primero proyecto/run.py.")
        return 1
    scale = min(1.0, 1024 / max(image.size))
    size = (round(image.width * scale), round(image.height * scale))
    image = image.resize(size, Image.BILINEAR)  # como prepare()
    device = get_device()
    with torch.no_grad():
        pred = detector.to(device)([to_tensor(image).to(device)])[0]
    boxes = pred["boxes"][pred["scores"] >= args.min_score].cpu().numpy()
    if len(boxes) == 0:
        print("No se detectaron mazorcas en la foto: no hay diagnóstico.")
        return 0
    crops = [crop_pod(image, b, 224) for b in boxes]
    ds = CropDataset(crops, None, eval_transform())
    proba = predict_proba(classifier, ds, device)
    sick = 1 - proba[:, 0]
    fig, axes = plt.subplots(1, 1 + min(4, len(crops)), figsize=(4 + 2.5 * min(4, len(crops)), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    for i, (box, p, s) in enumerate(zip(boxes, proba, sick, strict=True)):
        stage = CLASSES[int(p.argmax())]
        alarm = s > threshold
        print(
            f"mazorca {i + 1}: {STAGE_NAMES[stage]} · P(enferma) = {s:.2f}"
            + (" · ALERTA" if alarm else "")
        )
        x1, y1, x2, y2 = box
        axes[0].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                lw=2,
                color="tab:red" if alarm else "tab:green",
            )
        )
        axes[0].text(x1, y1, str(i + 1), color="white", fontsize=9, backgroundcolor="black")
        if i < 4:
            x, _ = ds[i]
            cam = grad_cam(classifier, classifier.layer4, x[None].to(device), int(p.argmax()))[0]
            axes[i + 1].imshow(crops[i])
            axes[i + 1].imshow(cam.cpu().numpy(), cmap="inferno", alpha=0.45)
            axes[i + 1].set_title(f"{i + 1}: {stage} ({s:.2f})", fontsize=8)
    for a in axes:
        a.axis("off")
    out = args.salida or OUT / f"diagnostico_{path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
