"""Mini-proyecto: detección temprana de monilia en mazorcas de cacao (CocoaMoniliaDataSet).

Detector de mazorcas (Faster R-CNN) → clasificador de etapa (h0/m1/m2/m3) sobre cada recorte.
Partición principal por ráfaga de fotos, comparada con la publicada; control del celular; tres
hipótesis con bootstrap por ráfaga y Holm. Métrica central: recall de m1 con 10 % de falsas alarmas
en sanas.

Uso:
    uv sync --extra base --extra dl
    uv run python 02-semestre-2/01-vision-por-computador/data/download.py
    uv run python 02-semestre-2/01-vision-por-computador/proyecto/run.py
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from applied_cv import MODULE
from applied_cv.boxes import average_precision, average_precision_range
from applied_cv.cocoa.classifiers import (
    CropDataset,
    build_resnet,
    eval_transform,
    extract_features,
    fine_tune,
    fit_logistic,
    predict_proba,
    train_augmentation,
)
from applied_cv.cocoa.data import CLASSES
from applied_cv.cocoa.detector import DetectionDataset, build_detector, detect, train_detector
from applied_cv.cocoa.evaluation import (
    bootstrap_diff,
    bootstrap_unpaired,
    decide,
    holm,
    macro_f1,
    match_boxes,
    recall_above,
    sick_score,
    threshold_at_fpr,
)
from applied_cv.cocoa.features import color_texture_features
from applied_cv.cocoa.prepare import square_crop_box
from applied_cv.cocoa.splits import assign_bursts, burst_split, published_split, read_split_lists
from applied_cv.gradcam import grad_cam
from applied_dl.training import get_device
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import data_dir, save_fig, set_seed, setup_style

OUT = Path(__file__).resolve().parent / "resultados"
FULL = {
    "pretrained": True,
    "clf_epochs": 15,
    "clf_lr": 3e-4,
    "det_epochs": 10,
    "det_lr": 0.005,
    "det_min_size": 512,
    "det_max_size": 640,
    "batch": 32,
    "n_boot": 2000,
    "crop_size": 224,
}
SMOKE = {
    **FULL,
    "pretrained": False,
    "clf_epochs": 1,
    "clf_lr": 1e-3,
    "det_epochs": 1,
    "det_min_size": 128,
    "det_max_size": 160,
    "batch": 8,
    "n_boot": 50,
    "crop_size": 64,
}
PARTS = ("train", "val", "test")


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _jsonable(value):
    """Enteros y flotantes de NumPy → tipos de Python (conteos como enteros, no 747.0)."""
    return value.item() if isinstance(value, np.generic) else str(value)


def load_tables(prepared: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = pd.read_csv(prepared / "images.csv", parse_dates=["timestamp"])
    pods = pd.read_csv(prepared / "pods.csv")
    images["burst"] = assign_bursts(images)
    return images, pods


def stage_metrics(y_val, proba_val, y_test, proba_test) -> dict:
    """Umbral en validación (FPR 10 % en sanas) y métricas en test."""
    threshold = threshold_at_fpr(sick_score(proba_val)[y_val == 0], fpr=0.1)
    s = sick_score(proba_test)
    return {
        "umbral": threshold,
        "recall_m1": recall_above(s[y_test == 1], threshold),
        "fpr_h0": recall_above(s[y_test == 0], threshold),
        "f1_macro": macro_f1(y_test, proba_test),
        "confusion": confusion_matrix(y_test, proba_test.argmax(1), labels=range(4)).tolist(),
        "n_test": {c: int((y_test == k).sum()) for k, c in enumerate(CLASSES)},
    }


def train_stage_models(pods, X, F, paths, part, cfg, device) -> tuple[dict, torch.nn.Module]:
    """Los cuatro clasificadores con la partición `part`; probabilidades en val y test."""
    y = pods.label.map(CLASSES.index).to_numpy()
    idx = {p: np.flatnonzero(part == p) for p in PARTS}
    tr, va = idx["train"], idx["val"]
    prevalence = np.bincount(y[tr], minlength=4) / len(tr)
    color = fit_logistic(X[tr], y[tr], X[va], y[va])
    probe = fit_logistic(F[tr], y[tr], F[va], y[va])
    torch.manual_seed(42)
    model = build_resnet(4, cfg["pretrained"])
    fine_tune(
        model,
        CropDataset([paths[i] for i in tr], y[tr].tolist(), train_augmentation()),
        CropDataset([paths[i] for i in va], y[va].tolist(), eval_transform()),
        epochs=cfg["clf_epochs"],
        lr=cfg["clf_lr"],
        device=device,
        batch_size=cfg["batch"],
    )
    proba = {
        "prevalencia": {p: np.tile(prevalence, (len(idx[p]), 1)) for p in ("val", "test")},
        "color_textura": {p: color.predict_proba(X[idx[p]]) for p in ("val", "test")},
        "sonda_lineal": {p: probe.predict_proba(F[idx[p]]) for p in ("val", "test")},
        "fine_tuning": {
            p: predict_proba(
                model, CropDataset([paths[i] for i in idx[p]], None, eval_transform()), device
            )
            for p in ("val", "test")
        },
    }
    return proba, model


def camera_probe(F, pods_h0, n_boot) -> dict:
    """H3: dentro de las sanas, ¿se adivina el celular (IMG vs. epoch) con rasgos de ImageNet?"""
    y = (pods_h0.camera == "epoch").to_numpy().astype(int)
    groups = pods_h0.burst.to_numpy()
    n_splits = min(5, *(len(set(groups[y == k])) for k in (0, 1)))
    counts = {"IMG": int((y == 0).sum()), "epoch": int((y == 1).sum())}
    if n_splits < 2:  # un celular en una sola ráfaga: no hay validación cruzada agrupada posible
        nan = float("nan")
        return {
            "diff": nan,
            "ci": [nan, nan],
            "p_value": 1.0,
            "n_boot": 0,
            "exactitud_balanceada": nan,
            "n": counts,
            "nota": "no evaluable",
        }
    oof = np.zeros(len(y), dtype=int)
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
        F, y, groups
    ):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(F[tr], y[tr])
        oof[te] = clf.predict(F[te])

    def stat(i):
        if len(set(y[i])) < 2:
            return np.nan
        return balanced_accuracy_score(y[i], oof[i]) - 0.5

    res = bootstrap_diff(stat, groups, n_boot=n_boot)
    return {**res, "exactitud_balanceada": res["diff"] + 0.5, "n": counts}


def detection_and_system(images, pods, prepared, part_img, clf, threshold, cfg, device):
    """Detector entrenado en train (por ráfaga), AP en test y sistema completo en fotos de test."""
    boxes_of = {n: g[["x1", "y1", "x2", "y2"]].to_numpy() for n, g in pods.groupby("image")}
    train_names = images.image[part_img == "train"].tolist()
    test_names = images.image[part_img == "test"].tolist()
    detector = build_detector(cfg["pretrained"], cfg["det_min_size"], cfg["det_max_size"])
    losses = train_detector(
        detector,
        DetectionDataset(
            [prepared / "images" / n for n in train_names],
            [boxes_of[n] for n in train_names],
            flip_prob=0.5,
        ),
        epochs=cfg["det_epochs"],
        device=device,
        lr=cfg["det_lr"],
    )
    found = detect(detector, [prepared / "images" / n for n in test_names], device)
    gt = [boxes_of[n] for n in test_names]
    det = {
        "ap50": average_precision([b for b, _ in found], [s for _, s in found], gt, 0.5),
        "ap50_95": average_precision_range([b for b, _ in found], [s for _, s in found], gt),
        "perdidas": losses,
        "fotos_test": len(test_names),
    }
    rows = []
    for name, (boxes, scores) in zip(test_names, found, strict=True):
        keep = scores >= 0.5
        boxes = boxes[keep]
        image = Image.fromarray(_rgb(prepared / "images" / name))
        crops = [
            image.crop(square_crop_box(b, image.width, image.height)).resize(
                (cfg["crop_size"], cfg["crop_size"])
            )
            for b in boxes
        ]
        sick = (
            sick_score(predict_proba(clf, CropDataset(crops, None, eval_transform()), device))
            if crops
            else np.zeros(0)
        )
        truth = pods[pods.image == name]
        match = match_boxes(boxes, truth[["x1", "y1", "x2", "y2"]].to_numpy())
        for label, m in zip(truth.label, match, strict=True):
            rows.append((label, m >= 0, m >= 0 and sick[m] > threshold))
    sys_df = pd.DataFrame(rows, columns=["label", "detectada", "alerta"])
    by_label = sys_df.groupby("label")
    system = {
        "recall_m1": float(sys_df[sys_df.label == "m1"].alerta.mean()),
        "fpr_h0": float(sys_df[sys_df.label == "h0"].alerta.mean()),
        "detectadas_por_clase": by_label.detectada.mean().round(4).to_dict(),
        "mazorcas_test": by_label.size().to_dict(),
    }
    return det, system, detector


def gradcam_mosaic(model, pods_test, y_test, proba_test, threshold, paths, device, path):
    """Errores críticos: m1 sin alerta y sanas con alerta, con el mapa Grad-CAM del modelo."""
    s = sick_score(proba_test)
    missed = np.flatnonzero((y_test == 1) & (s <= threshold))[:6]
    false_alarm = np.flatnonzero((y_test == 0) & (s > threshold))[:6]
    chosen = [(i, "m1 sin alerta") for i in missed] + [(i, "sana con alerta") for i in false_alarm]
    if not chosen:
        return
    fig, axes = plt.subplots(2, len(chosen), figsize=(2.2 * len(chosen), 4.6), squeeze=False)
    model.to(device).eval()
    for j, (i, title) in enumerate(chosen):
        x, _ = CropDataset([paths[i]], None, eval_transform())[0]
        cam = grad_cam(model, model.layer4, x[None].to(device), int(proba_test[i].argmax()))[0]
        axes[0, j].imshow(_rgb(paths[i]))
        axes[0, j].set_title(f"{title}\nP(enf.)={s[i]:.2f}", fontsize=7)
        axes[1, j].imshow(_rgb(paths[i]))
        axes[1, j].imshow(cam.cpu().numpy(), cmap="inferno", alpha=0.5)
        for a in axes[:, j]:
            a.axis("off")
    save_fig(fig, path)
    plt.close(fig)


def run(data: Path, out: Path, *, smoke=False, device=None, models_dir=None) -> dict:
    cfg = SMOKE if smoke else FULL
    t0 = time.time()
    set_seed(42)
    device = device or get_device()
    out.mkdir(parents=True, exist_ok=True)
    models_dir = models_dir or data / "modelos"
    models_dir.mkdir(parents=True, exist_ok=True)
    prepared = data / "prepared"
    images, pods = load_tables(prepared)
    images["part_rafaga"] = burst_split(images, images.burst).to_numpy()
    images["part_publicada"], conflicts = published_split(images, read_split_lists(data))
    meta = images.set_index("image")[["burst", "camera", "part_rafaga", "part_publicada"]]
    all_pods = pods.join(meta, on="image")
    stage = all_pods[all_pods.label.isin(CLASSES)].reset_index(drop=True)
    y = stage.label.map(CLASSES.index).to_numpy()
    paths = [prepared / c for c in stage.crop]

    X = np.stack([color_texture_features(_rgb(p)) for p in paths])
    backbone = build_resnet(4, cfg["pretrained"])
    F = extract_features(backbone, CropDataset(paths, None, eval_transform()), device)

    report = {
        "datos": {
            "fotos": len(images),
            "mazorcas": len(all_pods),
            "por_etiqueta": all_pods.label.value_counts().to_dict(),
            "m4_excluidas": int((all_pods.label == "m4").sum()),
            "rafagas": int(images.burst.nunique()),
            "camaras": images.camera.value_counts().to_dict(),
            "conflictos_publicada": conflicts,
            "fuera_publicada": int((images.part_publicada == "fuera").sum()),
            "fotos_por_parte_rafaga": images.part_rafaga.value_counts().to_dict(),
        },
        "config": cfg,
    }

    # 1. Partición por ráfaga: los cuatro clasificadores.
    part = stage.part_rafaga.to_numpy()
    proba, clf = train_stage_models(stage, X, F, paths, part, cfg, device)
    yv, yt = y[part == "val"], y[part == "test"]
    report["clasificacion_rafaga"] = {
        name: stage_metrics(yv, p["val"], yt, p["test"]) for name, p in proba.items()
    }
    test = stage[part == "test"].reset_index(drop=True)
    bursts_test = test.burst.to_numpy()

    # H1: fine-tuning − color+textura en recall de m1 (mismas mazorcas, pareado).
    s_ft, s_ct = (
        sick_score(proba["fine_tuning"]["test"]),
        sick_score(proba["color_textura"]["test"]),
    )
    thr_ft = report["clasificacion_rafaga"]["fine_tuning"]["umbral"]
    thr_ct = report["clasificacion_rafaga"]["color_textura"]["umbral"]

    def recall_gap(i):
        m = yt[i] == 1
        return np.mean(s_ft[i][m] > thr_ft) - np.mean(s_ct[i][m] > thr_ct) if m.any() else np.nan

    h1 = bootstrap_diff(recall_gap, bursts_test, n_boot=cfg["n_boot"])

    # 2. Partición publicada (misma receta de fine-tuning) y H2.
    pub = stage.part_publicada.to_numpy()
    usable = pub != "fuera"
    proba_pub, _ = train_stage_models(
        stage[usable].reset_index(drop=True),
        X[usable],
        F[usable],
        [p for p, u in zip(paths, usable, strict=True) if u],
        pub[usable],
        cfg,
        device,
    )
    y_pub = y[usable]
    yv_pub, yt_pub = y_pub[pub[usable] == "val"], y_pub[pub[usable] == "test"]
    report["clasificacion_publicada"] = {
        "fine_tuning": stage_metrics(
            yv_pub, proba_pub["fine_tuning"]["val"], yt_pub, proba_pub["fine_tuning"]["test"]
        )
    }
    bursts_pub = stage[usable].burst.to_numpy()[pub[usable] == "test"]
    p_pub, p_raf = proba_pub["fine_tuning"]["test"], proba["fine_tuning"]["test"]
    h2 = bootstrap_unpaired(
        lambda i: macro_f1(yt_pub[i], p_pub[i]),
        bursts_pub,
        lambda i: macro_f1(yt[i], p_raf[i]),
        bursts_test,
        n_boot=cfg["n_boot"],
    )

    # 3. Control de celular: el mismo modelo y umbral, por formato de celular.
    report["control_celular"] = {
        camera: {
            "recall_m1": recall_above(s_ft[(test.camera == camera) & (yt == 1)], thr_ft),
            "fpr_h0": recall_above(s_ft[(test.camera == camera) & (yt == 0)], thr_ft),
            "n_m1": int(((test.camera == camera) & (yt == 1)).sum()),
            "n_h0": int(((test.camera == camera) & (yt == 0)).sum()),
        }
        for camera in sorted(test.camera.unique())
    }

    # H3: huella del celular dentro de las sanas.
    h0_mask = (y == 0) & stage.camera.isin(["IMG", "epoch"]).to_numpy()
    h3 = camera_probe(F[h0_mask], stage[h0_mask], cfg["n_boot"])
    report["sonda_celular"] = {
        "exactitud_balanceada": h3.pop("exactitud_balanceada"),
        "n": h3.pop("n"),
    }

    adjusted = holm({"H1": h1["p_value"], "H2": h2["p_value"], "H3": h3["p_value"]})
    report["hipotesis"] = {
        "H1": decide(h1, adjusted["H1"]),
        "H2": decide(h2, adjusted["H2"]),
        "H3": decide(h3, adjusted["H3"]),
    }

    # 4. Detector y sistema completo.
    report["detector"], report["sistema"], detector = detection_and_system(
        images, all_pods, prepared, images.part_rafaga, clf, thr_ft, cfg, device
    )

    # Figuras y modelos para la demo.
    setup_style()
    cm = np.array(report["clasificacion_rafaga"]["fine_tuning"]["confusion"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, v, ha="center", va="center", fontsize=8)
    ax.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="predicha",
        ylabel="real",
        title="Fine-tuning · test por ráfaga",
    )
    save_fig(fig, out / "confusion.png")
    plt.close(fig)
    names = list(report["clasificacion_rafaga"])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, [report["clasificacion_rafaga"][n]["recall_m1"] for n in names])
    ax.set(ylabel="recall de m1 (FPR h0 = 10 % en val)", ylim=(0, 1))
    save_fig(fig, out / "recall_m1.png")
    plt.close(fig)
    test_paths = [p for p, k in zip(paths, part, strict=True) if k == "test"]
    gradcam_mosaic(
        clf,
        test,
        yt,
        proba["fine_tuning"]["test"],
        thr_ft,
        test_paths,
        device,
        out / "errores_gradcam.png",
    )
    torch.save(clf.cpu().state_dict(), models_dir / "clasificador.pt")
    torch.save(detector.cpu().state_dict(), models_dir / "detector.pt")
    (models_dir / "modelos.json").write_text(
        json.dumps(
            {
                "umbral": thr_ft,
                "det_min_size": cfg["det_min_size"],
                "det_max_size": cfg["det_max_size"],
            }
        )
    )
    report["segundos"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_jsonable)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    result = run(data_dir(MODULE), OUT)
    for name, h in result["hipotesis"].items():
        print(
            f"{name}: Δ = {h['diff']:+.4f} IC [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] "
            f"p_Holm = {h['p_holm']:.4f} → apoya: {h['apoya_hipotesis']}"
        )
