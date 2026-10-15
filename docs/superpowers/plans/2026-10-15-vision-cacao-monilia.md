# Visión por computador · Detección temprana de monilia en cacao — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el módulo S2 «Visión por computador» (paquete `applied_cv`, ejercicios, 5 notebooks
smoke) y su mini-proyecto: detector de mazorcas + clasificador de etapa de monilia sobre
CocoaMoniliaDataSet, con evaluación honesta (particiones por ráfaga, control de celular) y tres
hipótesis.

**Architecture:** `applied_cv` reúne las piezas desde cero (NumPy: filtros, HSV, LBP, IoU, NMS, AP,
Dice) y las de PyTorch (U-Net, Grad-CAM). El subpaquete `applied_cv.cocoa` contiene lo específico del
proyecto (anotaciones COCO, preparación, ráfagas y particiones, rasgos, clasificadores, detector y
evaluación). `proyecto/run.py` orquesta los experimentos y `proyecto/diagnosticar.py` es la demo.

**Tech Stack:** Python 3.12, NumPy, pandas, scikit-learn, PyTorch + torchvision (Faster R-CNN
MobileNetV3-FPN, ResNet-18), Pillow, scikit-image solo como referencia en tests; uv, ruff, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-vision-cacao-monilia-design.md`

**Módulo:** `M = 02-semestre-2/01-vision-por-computador` (todas las rutas relativas a la raíz del repo).

## Global Constraints

- Idioma: español para docs y comentarios; nombres de código en inglés (CLAUDE.md).
- Gestor uv; nunca pip. Antes de cada commit: `uv run ruff format && uv run ruff check --fix && uv run pytest`
  (formatear antes de chequear; nunca encadenar `pytest | tail` antes de `git commit`).
- Línea máxima 100 (ruff). Nunca `sys.path.insert`.
- Aleatoriedad: semilla 42; `rng = common.set_seed(42)` en notebooks; generadores explícitos.
- Datos y pesos nunca al repo (`**/data/*` ignorado); `data/download.py` descarga y verifica checksums.
- El CI instala `--extra base --extra dl`, no descarga datos ni pesos preentrenados; notebooks smoke
  (`"smoke": true`) corren en segundos sin descargas.
- Dataset: CocoaMoniliaDataSet v2, Zenodo 17716661, CC BY 4.0; zip 6 187 303 002 bytes, md5
  `d3531c4a62f3aaff9c146c7a8ae39a6b`.
- Nada de `ultralytics`/YOLO (licencia AGPL; el repo es MIT y público).
- Aumentación del clasificador: nunca cambia el tono (el color es el síntoma).
- Métrica central: recall de m1 con FPR en h0 = 10 %, puntaje P(enferma) = 1 − P(h0), umbral en val.
- Hipótesis: bootstrap por ráfaga, 2 000 réplicas, IC 95 %, Holm sobre H1–H3; «apoya» = rechaza ∧
  dirección prevista.
- Trabajo en la rama `vision-cacao` (no en main); preguntar al usuario antes del merge.

## Hechos verificados antes de escribir el plan (2026-10-15)

Leídos del zip remoto (peticiones HTTP Range) y prototipados en un borrador:

- El zip trae `CocoaMoniliaDataSet/COCO_annotations/instances_{h0,m1,m2,m3}.json`,
  `cocoapods_images/<carpeta>/` (646/436/401/467 = 1 950 archivos), `YOLO_annotations/`,
  `mask_segmentation/`.
- Cada JSON reinicia sus ids de imagen en 1; categorías `1 h0, 2 m1, 3 m2, 4 m3, 5 m4` (m4 no está
  documentada). Una carpeta puede contener mazorcas de otra clase (la etiqueta es por anotación).
- 1 953 entradas de imagen en los JSON, **1 913 nombres únicos** (40 fotos están en dos carpetas);
  2 162 anotaciones → **2 121 mazorcas** tras quitar duplicados (IoU > 0,9 en la misma foto):
  h0 747, m1 439, m2 403, m3 494, m4 38.
- `bbox` COCO = [x, y, w, h] en píxeles de la foto original (p. ej. 3472×3472); `date_captured` = 0.
- Formatos de nombre: `IMG_AAAAMMDD_HHMMSS…`, `AAAAMMDD_HHMMSS`, epoch en ms (13 dígitos, todas del
  2024-10-11), `IMG_E####`/`IMG_####` (sin hora, numeradas), y 3 `m0 (NN).jpg`.
- Intervalos entre fotos consecutivas del mismo formato: cuantiles 50 % = 4 s, 90 % = 72 s,
  95 % = 134 s. Con Δ = 120 s salen 109 ráfagas con hora (+ las numeradas y sueltas).
- Faster R-CNN MobileNetV3-FPN (torchvision 0.29, pesos None) entrena en MPS a 0,42 s/lote de 4 con
  `min_size=512, max_size=640` (CPU: 2,0 s): ~2–3 min por época.
- Prototipos verificados: `conv2d` = `scipy.ndimage.correlate(mode="reflect")`; `rgb_to_hsv` =
  `skimage.color.rgb2hsv`; LBP = `skimage.feature.local_binary_pattern(P=8, R=1, "default")` bit a
  bit; `box_iou`/`nms` = torchvision; AP del caso a mano = 0,8333; Grad-CAM de la red de prueba =
  entrada/máximo; bilineal = `F.interpolate(align_corners=False)`; Otsu = `skimage.filters.threshold_otsu`;
  campo receptivo teórico = empírico.

## Interfaces (resumen)

```
applied_cv.image:   conv2d(img, k) · sobel(img) · rgb_to_hsv(rgb01) · rgb_to_gray(rgb01)
                    local_binary_pattern(gray) · color_histogram(rgb01, bins) · lbp_histogram(gray255)
applied_cv.boxes:   box_iou(a, b) · nms(boxes, scores, thr) · average_precision(pb, ps, gt, thr)
                    average_precision_range(pb, ps, gt)
applied_cv.masks:   dice(pred, target) · mask_iou(pred, target)
applied_cv.unet:    UNet(in_channels, n_classes, base)
applied_cv.gradcam: grad_cam(model, layer, x, target) -> Tensor[N, H, W] en [0, 1]
applied_cv.cocoa.data:        CLASSES · parse_filename(name) · load_annotations(root) -> (images, pods)
applied_cv.cocoa.prepare:     square_crop_box(box, w, h, margin) · prepare(raw_root, out_dir, ...) -> dict
applied_cv.cocoa.splits:      assign_bursts(images, gap_seconds, seq_gap) · burst_split(images, bursts, seed)
                              read_split_lists(directory) · published_split(images, lists)
applied_cv.cocoa.features:    color_texture_features(rgb_uint8)
applied_cv.cocoa.classifiers: CropDataset · eval_transform() · train_augmentation() · build_resnet()
                              fine_tune() · predict_proba() · extract_features() · fit_logistic()
applied_cv.cocoa.evaluation:  sick_score · threshold_at_fpr · recall_above · macro_f1
                              cluster_indices · bootstrap_diff · bootstrap_unpaired · holm · decide
                              match_boxes
applied_cv.cocoa.detector:    build_detector(pretrained) · DetectionDataset · collate
                              train_detector() · detect()
```

## Review Focus

1. **Orientación EXIF**: fotos de celular con EXIF ≠ 1 podrían dejar las cajas desalineadas; se espera
   que `prepare` informe las orientaciones y que la Task 12 verifique con un mosaico qué convención
   alinea las cajas antes de entrenar. (Test: `test_prepare_reports_exif_and_size_mismatch`, Task 6.)
2. **Foto repetida en dos carpetas y en dos partes de la partición publicada**: no debe evaluarse en
   test lo que está en train; se espera exclusión y conteo. (Test: `test_published_split_conflicts`,
   Task 7.)
3. **Mazorcas `m4`**: se espera que el clasificador las excluya (y se reporte cuántas) y que el
   detector sí las use como mazorcas. (Test: `test_run_smoke_counts_m4`, Task 11.)
4. **Imagen en el JSON sin archivo**: se espera exclusión con aviso, nunca un error a mitad del
   preprocesamiento. (Test: `test_prepare_resizes_crops_and_reports_missing`, Task 6.)
5. **Demo con entrada inválida** (archivo inexistente, archivo que no es imagen, foto sin mazorcas):
   mensaje claro, código de salida ≠ 0 para errores y ningún diagnóstico inventado. (Tests en Task 11.)

---

### Task 1: Paquete `applied_cv` e `image.py`

**Files:**
- Modify: `pyproject.toml` (extra `dl` y `packages`)
- Create: `M/src/applied_cv/__init__.py`, `M/src/applied_cv/image.py`
- Test: `M/tests/test_image.py`
- Delete: `M/src/.gitkeep`, `M/tests/.gitkeep`

**Interfaces:**
- Produces: `conv2d(image: ndarray, kernel: ndarray) -> ndarray`; `sobel(image) -> ndarray`;
  `rgb_to_hsv(rgb: float [0,1]) -> ndarray`; `rgb_to_gray(rgb: float [0,1]) -> ndarray`;
  `local_binary_pattern(gray) -> ndarray[int64]`; `color_histogram(rgb01, bins=16) -> ndarray[3·bins]`;
  `lbp_histogram(gray255) -> ndarray[256]`; `applied_cv.MODULE: str`.

- [ ] **Step 1: Crear la rama y registrar el paquete**

```bash
git checkout -b vision-cacao
git rm -q 02-semestre-2/01-vision-por-computador/src/.gitkeep 02-semestre-2/01-vision-por-computador/tests/.gitkeep
```

En `pyproject.toml`, cambiar la línea del extra `dl` y agregar el paquete al final de `packages`:

```toml
dl = ["torch>=2.5", "torchvision>=0.20", "scikit-image>=0.24"]  # skimage: referencia en tests de visión
```

```toml
    "01-semestre-1/proyecto-integrador-s1/src/book_topics",
    "02-semestre-2/01-vision-por-computador/src/applied_cv",
]
```

Run: `uv lock && uv sync --extra base --extra dl`
Expected: instala scikit-image; sin errores.

- [ ] **Step 2: Escribir los tests (fallan)**

`M/tests/test_image.py`:

```python
import numpy as np
import pytest
from applied_cv.image import (
    color_histogram,
    conv2d,
    lbp_histogram,
    local_binary_pattern,
    rgb_to_gray,
    rgb_to_hsv,
    sobel,
)
from scipy import ndimage as ndi

RNG = np.random.default_rng(0)


def test_conv2d_matches_scipy_correlate():
    img, k = RNG.random((13, 17)), RNG.random((3, 5))
    np.testing.assert_allclose(conv2d(img, k), ndi.correlate(img, k, mode="reflect"), atol=1e-12)


def test_sobel_matches_scipy():
    img = RNG.random((10, 12))
    ref = np.hypot(ndi.sobel(img, axis=1, mode="reflect"), ndi.sobel(img, axis=0, mode="reflect"))
    np.testing.assert_allclose(sobel(img), ref, atol=1e-12)


def test_rgb_to_hsv_matches_skimage_including_gray_and_black():
    color = pytest.importorskip("skimage.color")
    rgb = RNG.random((20, 20, 3))
    rgb[0, 0], rgb[0, 1] = 0.5, 0.0  # gris (sin tono) y negro (sin saturación)
    np.testing.assert_allclose(rgb_to_hsv(rgb), color.rgb2hsv(rgb), atol=1e-12)


def test_rgb_to_gray_matches_skimage():
    color = pytest.importorskip("skimage.color")
    rgb = RNG.random((8, 9, 3))
    np.testing.assert_allclose(rgb_to_gray(rgb), color.rgb2gray(rgb), atol=1e-12)


def test_lbp_matches_skimage_bit_for_bit():
    feature = pytest.importorskip("skimage.feature")
    gray = (RNG.random((15, 12)) * 255).round().astype(np.uint8)
    ref = feature.local_binary_pattern(gray, 8, 1, method="default")
    np.testing.assert_array_equal(local_binary_pattern(gray), ref)


def test_histograms_are_normalized():
    rgb = RNG.random((16, 16, 3))
    hist = color_histogram(rgb, bins=8)
    assert hist.shape == (24,)
    np.testing.assert_allclose(hist.reshape(3, 8).sum(axis=1), 1.0)
    lbp = lbp_histogram(np.round(rgb_to_gray(rgb) * 255))
    assert lbp.shape == (256,) and np.isclose(lbp.sum(), 1.0)
```

- [ ] **Step 3: Ver que fallan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_image.py -q`
Expected: ERROR de colección `ModuleNotFoundError: No module named 'applied_cv'` (o `applied_cv.image`).

- [ ] **Step 4: Implementar**

`M/src/applied_cv/__init__.py`:

```python
"""Visión por computador: operaciones clásicas, cajas, máscaras, U-Net y Grad-CAM.

`image`, `boxes` y `masks` solo usan NumPy (se verifican contra scikit-image y torchvision);
`unet`, `gradcam` y `cocoa` requieren el extra `dl`.
"""

MODULE = "02-semestre-2/01-vision-por-computador"
```

`M/src/applied_cv/image.py`:

```python
"""Operaciones clásicas sobre imágenes, desde cero con NumPy."""

import numpy as np

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
# Pesos de luminancia ITU-R 709, los mismos que skimage.color.rgb2gray.
GRAY_WEIGHTS = np.array([0.2125, 0.7154, 0.0721])


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlación 2D del mismo tamaño, con relleno por reflexión de borde.

    Equivale a `scipy.ndimage.correlate(mode="reflect")` (que NumPy llama "symmetric"). Las CNN
    también calculan correlaciones, aunque se las llame convoluciones.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(
        np.asarray(image, dtype=float), ((ph, kh - 1 - ph), (pw, kw - 1 - pw)), mode="symmetric"
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def sobel(image: np.ndarray) -> np.ndarray:
    """Magnitud del gradiente con los núcleos de Sobel (bordes)."""
    return np.hypot(conv2d(image, SOBEL_X), conv2d(image, SOBEL_X.T))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB en [0, 1] → HSV en [0, 1] (mismo convenio que skimage.color.rgb2hsv).

    El tono (H) separa «qué color» de «cuánta luz» (V): por eso el café de una lesión se reconoce
    mejor en H que en R, G y B, que cambian todos con la iluminación.
    """
    rgb = np.asarray(rgb, dtype=float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)
    delta = v - rgb.min(axis=-1)
    s = np.divide(delta, v, out=np.zeros_like(v), where=v > 0)
    safe = np.where(delta > 0, delta, 1.0)
    h = np.select(
        [delta == 0, v == r, v == g],
        [0.0, ((g - b) / safe) % 6, (b - r) / safe + 2],
        (r - g) / safe + 4,
    )
    return np.stack([h / 6, s, v], axis=-1)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luminancia en [0, 1] a partir de RGB en [0, 1]."""
    return np.asarray(rgb, dtype=float)[..., :3] @ GRAY_WEIGHTS


def local_binary_pattern(gray: np.ndarray) -> np.ndarray:
    """LBP de 8 vecinos en radio 1 con interpolación bilineal (skimage, method="default").

    Cada vecino sobre el círculo aporta un bit: 1 si es ≥ que el pixel central. El código de 8 bits
    describe la micro-textura local (borde, esquina, punto, zona plana) sin depender del brillo.
    Fuera de la imagen el valor es 0, como en skimage.
    """
    g = np.asarray(gray, dtype=float)
    h, w = g.shape
    padded = np.pad(g, 1, mode="constant")
    out = np.zeros((h, w), dtype=np.int64)
    rows, cols = np.mgrid[0:h, 0:w]
    for p in range(8):
        angle = 2 * np.pi * p / 8
        dr, dc = round(-np.sin(angle), 5), round(np.cos(angle), 5)
        r, c = rows + 1 + dr, cols + 1 + dc
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        fr, fc = r - r0, c - c0
        r1, c1 = np.minimum(r0 + 1, h + 1), np.minimum(c0 + 1, w + 1)
        value = (
            padded[r0, c0] * (1 - fr) * (1 - fc)
            + padded[r0, c1] * (1 - fr) * fc
            + padded[r1, c0] * fr * (1 - fc)
            + padded[r1, c1] * fr * fc
        )
        out |= (value >= g).astype(np.int64) << p
    return out


def color_histogram(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    """Histogramas normalizados de H, S y V concatenados (3·bins valores)."""
    hsv = rgb_to_hsv(rgb).reshape(-1, 3)
    return np.concatenate(
        [np.histogram(hsv[:, i], bins=bins, range=(0.0, 1.0))[0] / len(hsv) for i in range(3)]
    )


def lbp_histogram(gray255: np.ndarray) -> np.ndarray:
    """Histograma normalizado de los 256 códigos LBP de una imagen gris en [0, 255]."""
    codes = local_binary_pattern(gray255)
    return np.bincount(codes.ravel(), minlength=256) / codes.size
```

- [ ] **Step 5: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_image.py -q`
Expected: `6 passed`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add pyproject.toml uv.lock 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): paquete applied_cv con filtros, HSV y LBP desde cero"
```

---

### Task 2: Cajas (IoU, NMS, AP) y máscaras (Dice, IoU)

**Files:**
- Create: `M/src/applied_cv/boxes.py`, `M/src/applied_cv/masks.py`
- Test: `M/tests/test_boxes.py`, `M/tests/test_masks.py`

**Interfaces:**
- Produces: `box_iou(a, b) -> ndarray[len(a), len(b)]` (cajas `[x1, y1, x2, y2]`);
  `nms(boxes, scores, iou_threshold) -> ndarray[int]`;
  `average_precision(pred_boxes: list[ndarray], pred_scores: list[ndarray], gt_boxes: list[ndarray], iou_threshold=0.5) -> float`
  (nan si no hay cajas reales, 0.0 si no hay predicciones);
  `average_precision_range(pred_boxes, pred_scores, gt_boxes) -> float` (AP50:95);
  `dice(pred, target) -> float`; `mask_iou(pred, target) -> float` (ambas 1.0 si las dos máscaras
  están vacías).

- [ ] **Step 1: Escribir los tests (fallan)**

`M/tests/test_boxes.py`:

```python
import numpy as np
import pytest
from applied_cv.boxes import average_precision, average_precision_range, box_iou, nms

torch = pytest.importorskip("torch")
ops = pytest.importorskip("torchvision.ops")
RNG = np.random.default_rng(0)


def _random_boxes(n):
    xy = RNG.random((n, 2)) * 50
    return np.c_[xy, xy + RNG.random((n, 2)) * 30 + 1]


def test_box_iou_matches_torchvision():
    a, b = _random_boxes(7), _random_boxes(5)
    ref = ops.box_iou(torch.tensor(a), torch.tensor(b)).numpy()
    np.testing.assert_allclose(box_iou(a, b), ref, atol=1e-12)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
def test_nms_matches_torchvision(threshold):
    boxes, scores = _random_boxes(40), RNG.random(40)
    ref = ops.nms(torch.tensor(boxes), torch.tensor(scores), threshold).numpy()
    np.testing.assert_array_equal(nms(boxes, scores, threshold), ref)


def test_average_precision_hand_computed():
    # 2 cajas reales; predicciones 0.9 (TP), 0.8 (FP), 0.7 (TP):
    # precisión 1, 1/2, 2/3; recall 1/2, 1/2, 1; envolvente 1, 2/3, 2/3 → AP = ½·1 + ½·⅔
    gt = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]])]
    pred = [np.array([[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 31]])]
    ap = average_precision(pred, [np.array([0.9, 0.8, 0.7])], gt)
    assert ap == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_average_precision_duplicates_and_edge_cases():
    gt = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]])]
    duplicate = [np.array([[0, 0, 10, 10], [0, 0, 10, 10]])]
    assert average_precision(duplicate, [np.array([0.9, 0.8])], gt) == pytest.approx(0.5)
    assert average_precision([gt[0]], [np.array([1.0, 0.5])], gt) == pytest.approx(1.0)
    assert average_precision_range([gt[0]], [np.array([1.0, 0.5])], gt) == pytest.approx(1.0)
    assert average_precision([np.zeros((0, 4))], [np.zeros(0)], gt) == 0.0
    assert np.isnan(average_precision([gt[0]], [np.array([1.0, 0.5])], [np.zeros((0, 4))]))
```

`M/tests/test_masks.py`:

```python
import numpy as np
import pytest
from applied_cv.masks import dice, mask_iou


def test_dice_and_iou_known_masks():
    a = np.zeros((4, 4), bool)
    b = np.zeros((4, 4), bool)
    a[:2] = True  # 8 pixeles
    b[1:3] = True  # 8 pixeles, 4 en común
    assert dice(a, b) == pytest.approx(2 * 4 / 16)
    assert mask_iou(a, b) == pytest.approx(4 / 12)


def test_empty_masks_agree_perfectly():
    empty = np.zeros((3, 3), bool)
    assert dice(empty, empty) == 1.0 and mask_iou(empty, empty) == 1.0
    full = np.ones((3, 3), bool)
    assert dice(empty, full) == 0.0 and mask_iou(empty, full) == 0.0
```

- [ ] **Step 2: Ver que fallan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_boxes.py 02-semestre-2/01-vision-por-computador/tests/test_masks.py -q`
Expected: ERROR `No module named 'applied_cv.boxes'` / `'applied_cv.masks'`.

- [ ] **Step 3: Implementar**

`M/src/applied_cv/boxes.py`:

```python
"""Cajas [x1, y1, x2, y2]: IoU, supresión de no máximos (NMS) y precisión promedio (AP)."""

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matriz IoU [len(a), len(b)]: intersección sobre unión de cada par de cajas."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(axis=-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(axis=-1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Índices conservados, de mayor a menor puntaje (mismo resultado que torchvision.ops.nms).

    Toma la caja de mayor puntaje, descarta las que se le superponen más que el umbral y repite: un
    detector propone muchas cajas casi iguales por objeto y NMS deja una.
    """
    boxes = np.asarray(boxes, dtype=float)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = rest[box_iou(boxes[i], boxes[rest])[0] <= iou_threshold]
    return np.array(keep, dtype=int)


def average_precision(
    pred_boxes: list[np.ndarray],
    pred_scores: list[np.ndarray],
    gt_boxes: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """AP de una clase sobre varias imágenes: área bajo la envolvente de precisión-recall.

    Las predicciones de todas las imágenes se ordenan por puntaje. Cada una se empareja con la caja
    real libre de mayor IoU de su imagen; si ese IoU ≥ umbral es verdadero positivo y la caja real
    queda usada (una segunda detección del mismo objeto es falso positivo).
    """
    n_gt = sum(len(np.asarray(g).reshape(-1, 4)) for g in gt_boxes)
    if n_gt == 0:
        return float("nan")
    records = [
        (score, img, j)
        for img, scores in enumerate(pred_scores)
        for j, score in enumerate(np.asarray(scores))
    ]
    if not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    gts = [np.asarray(g, dtype=float).reshape(-1, 4) for g in gt_boxes]
    used = [np.zeros(len(g), dtype=bool) for g in gts]
    tp = np.zeros(len(records))
    for k, (_, img, j) in enumerate(records):
        if len(gts[img]) == 0:
            continue
        ious = box_iou(np.asarray(pred_boxes[img])[j], gts[img])[0]
        ious[used[img]] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            used[img][best] = True
            tp[k] = 1.0
    hits = np.cumsum(tp)
    recall = hits / n_gt
    precision = hits / np.arange(1, len(records) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float((np.diff(np.r_[0.0, recall]) * envelope).sum())


def average_precision_range(
    pred_boxes: list[np.ndarray], pred_scores: list[np.ndarray], gt_boxes: list[np.ndarray]
) -> float:
    """AP50:95: promedio del AP con umbrales de IoU 0.50, 0.55, …, 0.95 (convención COCO)."""
    return float(
        np.mean(
            [
                average_precision(pred_boxes, pred_scores, gt_boxes, t)
                for t in np.linspace(0.5, 0.95, 10)
            ]
        )
    )
```

`M/src/applied_cv/masks.py`:

```python
"""Métricas de segmentación sobre máscaras booleanas."""

import numpy as np


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    """2·|A∩B| / (|A| + |B|). Dos máscaras vacías coinciden perfectamente (1.0)."""
    pred, target = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    total = pred.sum() + target.sum()
    return 1.0 if total == 0 else float(2 * (pred & target).sum() / total)


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """|A∩B| / |A∪B|. Siempre ≤ Dice: IoU = Dice / (2 − Dice)."""
    pred, target = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    union = (pred | target).sum()
    return 1.0 if union == 0 else float((pred & target).sum() / union)
```

- [ ] **Step 4: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_boxes.py 02-semestre-2/01-vision-por-computador/tests/test_masks.py -q`
Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): IoU, NMS y AP de cajas; Dice e IoU de máscaras"
```

---

### Task 3: U-Net pequeña y Grad-CAM

**Files:**
- Create: `M/src/applied_cv/unet.py`, `M/src/applied_cv/gradcam.py`
- Test: `M/tests/test_unet.py`, `M/tests/test_gradcam.py`

**Interfaces:**
- Produces: `UNet(in_channels=3, n_classes=2, base=8)` (`nn.Module`; entrada [N, C, H, W] con H y W
  múltiplos de 4, salida [N, n_classes, H, W]; `ValueError` si no);
  `grad_cam(model, layer, x, target: int) -> Tensor[N, H, W]` en [0, 1].

- [ ] **Step 1: Escribir los tests (fallan)**

`M/tests/test_unet.py`:

```python
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
from applied_cv.unet import UNet  # noqa: E402


def test_unet_output_shape_and_size_check():
    net = UNet(3, 2, base=4)
    assert net(torch.zeros(2, 3, 32, 32)).shape == (2, 2, 32, 32)
    with pytest.raises(ValueError, match="múltiplos de 4"):
        net(torch.zeros(1, 3, 30, 32))


def test_unet_overfits_one_example():
    torch.manual_seed(0)
    net = UNet(1, 2, base=4)
    x = torch.rand(1, 1, 16, 16)
    y = (x[0, 0] > 0.5).long()[None]
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
```

`M/tests/test_gradcam.py`:

```python
import pytest

torch = pytest.importorskip("torch")
from applied_cv.gradcam import grad_cam  # noqa: E402


def test_grad_cam_of_known_network_is_the_normalized_input():
    # conv 1×1 con peso 1 → ReLU → promedio global → lineal con peso 1: el gradiente es uniforme,
    # así que Grad-CAM devuelve la activación (= la entrada) dividida por su máximo.
    nn = torch.nn
    model = nn.Sequential(
        nn.Conv2d(1, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 1)
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.zero_()
        model[4].weight.fill_(1.0)
        model[4].bias.zero_()
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 1:3, 2:5] = torch.tensor([[1.0, 2, 3], [4, 5, 6]])
    cam = grad_cam(model, model[0], x, target=0)
    torch.testing.assert_close(cam[0], x[0, 0] / 6)


def test_grad_cam_on_resnet_shape_and_range():
    models = pytest.importorskip("torchvision.models")
    net = models.resnet18(weights=None, num_classes=3).eval()
    cam = grad_cam(net, net.layer4, torch.rand(2, 3, 64, 64), target=1)
    assert cam.shape == (2, 64, 64)
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0 + 1e-6
```

- [ ] **Step 2: Ver que fallan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_unet.py 02-semestre-2/01-vision-por-computador/tests/test_gradcam.py -q`
Expected: ERROR `No module named 'applied_cv.unet'` / `'applied_cv.gradcam'`.

- [ ] **Step 3: Implementar**

`M/src/applied_cv/unet.py`:

```python
"""U-Net pequeña (Ronneberger et al., 2015) para segmentación: codificador, decodificador y saltos."""

import torch
from torch import nn


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Dos niveles de reducción. Las conexiones de salto devuelven al decodificador el detalle
    espacial que el codificador perdió al reducir: por eso los bordes de la máscara salen nítidos."""

    def __init__(self, in_channels: int = 3, n_classes: int = 2, base: int = 8):
        super().__init__()
        self.down1 = _double_conv(in_channels, base)
        self.down2 = _double_conv(base, 2 * base)
        self.bottom = _double_conv(2 * base, 4 * base)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = _double_conv(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = _double_conv(2 * base, base)
        self.head = nn.Conv2d(base, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError(f"H y W deben ser múltiplos de 4; llegó {tuple(x.shape[-2:])}")
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        b = self.bottom(self.pool(d2))
        u2 = self.dec2(torch.cat([self.up2(b), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.head(u1)
```

`M/src/applied_cv/gradcam.py`:

```python
"""Grad-CAM (Selvaraju et al., 2017): qué regiones de la imagen empujan la clase elegida."""

import torch
import torch.nn.functional as F
from torch import nn


def grad_cam(model: nn.Module, layer: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
    """Mapa [N, H, W] en [0, 1] para la clase `target`, al tamaño de la entrada.

    Pondera cada canal de las activaciones de `layer` por el promedio espacial de su gradiente,
    suma, aplica ReLU (solo evidencia a favor de la clase) y normaliza por imagen.
    """
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad()
        x = x.detach().clone().requires_grad_(True)  # garantiza el grafo aunque el modelo esté congelado
        model(x)[:, target].sum().backward()
    finally:
        handle.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-12)
    return (cam / peak[:, None, None]).detach()
```

- [ ] **Step 4: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_unet.py 02-semestre-2/01-vision-por-computador/tests/test_gradcam.py -q`
Expected: `4 passed`, sin advertencias de hooks.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): U-Net pequeña y Grad-CAM"
```

---
### Task 4: Ejercicios con soluciones

**Files:**
- Create: `M/ejercicios/README.md`, `M/ejercicios/ej01_bilineal.py` … `ej06_hsv_a_rgb.py` (enunciados),
  `M/ejercicios/soluciones/ej01_bilineal.py` … `ej06_hsv_a_rgb.py`
- Test: `M/tests/test_soluciones.py`
- Delete: `M/ejercicios/soluciones/.gitkeep`

**Interfaces:**
- Consumes: `applied_cv.boxes.box_iou`, `applied_cv.image.rgb_to_hsv`.
- Produces: cada archivo define su función y `check() -> None` (lanza `AssertionError` si falla).
  Los enunciados levantan `NotImplementedError`; las soluciones pasan su `check()`.

Patrón: el enunciado y la solución comparten **exactamente** el mismo `check()`; en el enunciado la
función solo tiene la firma, el docstring y `raise NotImplementedError`.

- [ ] **Step 1: Test de soluciones (falla: no hay soluciones)**

`M/tests/test_soluciones.py`:

```python
"""Ejecuta el `check()` de cada solución: garantiza que las soluciones publicadas son correctas."""

import importlib.util
from pathlib import Path

import pytest

SOLUTIONS = sorted((Path(__file__).parents[1] / "ejercicios" / "soluciones").glob("ej*.py"))


def test_there_are_six_solutions():
    assert len(SOLUTIONS) == 6


@pytest.mark.parametrize("path", SOLUTIONS, ids=lambda p: p.stem)
def test_solution_passes_its_check(path):
    pytest.importorskip("skimage")
    spec = importlib.util.spec_from_file_location(f"sol_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check()
```

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_soluciones.py -q`
Expected: FAIL `assert 0 == 6`.

- [ ] **Step 2: Soluciones**

`M/ejercicios/soluciones/ej01_bilineal.py`:

```python
"""Solución ejercicio 1: redimensionar con interpolación bilineal.

Cada pixel de salida mira el punto (i + 0.5)·escala − 0.5 de la entrada (centros de pixel
alineados, como `align_corners=False`), recortado a la imagen, y mezcla sus cuatro vecinos con
pesos proporcionales a la cercanía. Primero se interpola en columnas y luego en filas.
"""

import numpy as np


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = img.shape

    def coords(n_out: int, n_in: int):
        src = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        lo = np.floor(src).astype(int)
        return lo, np.minimum(lo + 1, n_in - 1), src - lo

    r0, r1, fr = coords(out_h, h)
    c0, c1, fc = coords(out_w, w)
    top = img[r0][:, c0] * (1 - fc) + img[r0][:, c1] * fc
    bottom = img[r1][:, c0] * (1 - fc) + img[r1][:, c1] * fc
    return top * (1 - fr)[:, None] + bottom * fr[:, None]


def check() -> None:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    for h, w, oh, ow in [(5, 7, 11, 13), (8, 8, 3, 5), (6, 4, 6, 4), (4, 9, 10, 2)]:
        x = rng.random((h, w))
        ref = F.interpolate(
            torch.tensor(x)[None, None], size=(oh, ow), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        np.testing.assert_allclose(resize_bilinear(x, oh, ow), ref, atol=1e-12)
    print("ej01 ok — coincide con torch.nn.functional.interpolate")


if __name__ == "__main__":
    check()
```

`M/ejercicios/soluciones/ej02_iou_bordes.py`:

```python
"""Solución ejercicio 2: IoU de dos cajas sin errores en los casos borde.

La intersección se recorta a 0 en cada eje (cajas separadas o que solo se tocan no se intersectan) y
si la unión es 0 (dos cajas sin área) se devuelve 0 en vez de dividir por cero.
"""

import numpy as np


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    def area(r) -> float:
        return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])

    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def check() -> None:
    import torch
    from torchvision.ops import box_iou

    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0, "idénticas"
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0, "separadas"
    assert iou((0, 0, 1, 1), (1, 0, 2, 1)) == 0.0, "solo se tocan en un borde"
    assert iou((0, 0, 4, 4), (1, 1, 2, 2)) == 1 / 16, "una dentro de otra"
    assert iou((1, 1, 1, 1), (1, 1, 1, 1)) == 0.0, "sin área: 0, no NaN"
    rng = np.random.default_rng(0)
    for _ in range(200):
        xy = rng.random((2, 2)) * 10
        boxes = np.c_[xy, xy + rng.random((2, 2)) * 5]
        ref = box_iou(torch.tensor(boxes[:1]), torch.tensor(boxes[1:])).item()
        assert abs(iou(*boxes) - ref) < 1e-12
    print("ej02 ok — casos borde y 200 pares aleatorios contra torchvision")


if __name__ == "__main__":
    check()
```

`M/ejercicios/soluciones/ej03_soft_nms.py`:

```python
"""Solución ejercicio 3: Soft-NMS gaussiano (Bodla et al., 2017).

En vez de borrar las cajas que se superponen con la elegida, se les baja el puntaje por
exp(−IoU²/σ). Dos mazorcas pegadas (IoU alto pero real) no desaparecen: solo bajan en el orden.
"""

import numpy as np
from applied_cv.boxes import box_iou


def soft_nms(
    boxes: np.ndarray, scores: np.ndarray, sigma: float = 0.5, score_threshold: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float).copy()
    remaining = list(range(len(scores)))
    keep, kept_scores = [], []
    while remaining:
        best = max(remaining, key=lambda i: scores[i])
        if scores[best] < score_threshold:
            break
        keep.append(best)
        kept_scores.append(scores[best])
        remaining.remove(best)
        if remaining:
            ious = box_iou(boxes[best], boxes[remaining])[0]
            scores[remaining] *= np.exp(-(ious**2) / sigma)
    return np.array(keep, dtype=int), np.array(kept_scores)


def check() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 20], [50, 50, 60, 60]])  # IoU(A, B) = 0.5
    keep, new_scores = soft_nms(boxes, np.array([0.9, 0.8, 0.7]), sigma=0.5)
    np.testing.assert_array_equal(keep, [0, 2, 1])
    np.testing.assert_allclose(new_scores, [0.9, 0.7, 0.8 * np.exp(-0.25 / 0.5)])
    keep, _ = soft_nms(boxes, np.array([0.9, 0.8, 0.7]), sigma=0.5, score_threshold=0.6)
    np.testing.assert_array_equal(keep, [0, 2])
    print("ej03 ok — B baja de 0.80 a 0.49 y queda detrás de C, pero no desaparece")


if __name__ == "__main__":
    check()
```

`M/ejercicios/soluciones/ej04_campo_receptivo.py`:

```python
"""Solución ejercicio 4: tamaño de salida y campo receptivo de una pila de convoluciones.

Salida: ⌊(n + 2p − d·(k − 1) − 1) / s⌋ + 1. Campo receptivo: cada capa agrega (k − 1)·d pixeles
multiplicados por el «salto» acumulado (producto de los strides anteriores).
"""

import numpy as np


def conv_output_size(n: int, k: int, s: int = 1, p: int = 0, d: int = 1) -> int:
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def receptive_field(layers: list[tuple[int, int, int]]) -> int:
    """`layers` = [(kernel, stride, dilation), ...] en orden de la red."""
    rf, jump = 1, 1
    for k, s, d in layers:
        rf += (k - 1) * d * jump
        jump *= s
    return rf


def check() -> None:
    import torch

    for n, k, s, p, d in [(32, 3, 1, 1, 1), (32, 3, 2, 1, 1), (31, 5, 2, 0, 1), (64, 3, 1, 2, 2)]:
        conv = torch.nn.Conv2d(1, 1, k, stride=s, padding=p, dilation=d)
        assert conv(torch.zeros(1, 1, n, n)).shape[-1] == conv_output_size(n, k, s, p, d)
    for layers in (
        [(3, 1, 1), (3, 1, 1)],
        [(3, 2, 1), (3, 1, 1)],
        [(7, 2, 1), (3, 2, 1), (3, 1, 1), (3, 1, 1)],
        [(3, 1, 2), (3, 1, 4)],
    ):
        convs = [torch.nn.Conv2d(1, 1, k, stride=s, dilation=d, bias=False) for k, s, d in layers]
        for c in convs:
            torch.nn.init.constant_(c.weight, 1.0)
        x = torch.ones(1, 1, 64, 64, requires_grad=True)
        y = torch.nn.Sequential(*convs)(x)
        y[0, 0, y.shape[2] // 2, y.shape[3] // 2].backward()
        rows = np.flatnonzero(x.grad[0, 0].sum(dim=1).numpy())
        assert rows.max() - rows.min() + 1 == receptive_field(layers), layers
    print("ej04 ok — fórmulas = tamaños de torch y campo receptivo medido con gradientes")


if __name__ == "__main__":
    check()
```

`M/ejercicios/soluciones/ej05_otsu.py`:

```python
"""Solución ejercicio 5: umbral de Otsu (1979).

Para cada umbral t se separan los pixeles en ≤ t y > t; Otsu elige el t que maximiza la varianza
entre las dos clases, w0·w1·(μ0 − μ1)², calculable para todos los t con sumas acumuladas.
"""

import numpy as np


def otsu_threshold(gray: np.ndarray) -> int:
    g = np.asarray(gray).astype(int)
    values = np.arange(g.min(), g.max() + 1)
    p = np.bincount((g - g.min()).ravel(), minlength=len(values)) / g.size
    w0 = np.cumsum(p)
    mu = np.cumsum(p * values)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    between = np.nan_to_num(between[:-1], nan=-1.0)
    return int(values[np.argmax(between)])


def check() -> None:
    from skimage.filters import threshold_otsu

    for seed in range(5):
        rng = np.random.default_rng(seed)
        pixels = np.r_[rng.normal(60, 15, 3000), rng.normal(170, 25, 2000)]
        gray = pixels.clip(0, 255).astype(np.uint8).reshape(50, 100)
        assert otsu_threshold(gray) == threshold_otsu(gray), seed
    print("ej05 ok — coincide con skimage.filters.threshold_otsu")


if __name__ == "__main__":
    check()
```

`M/ejercicios/soluciones/ej06_hsv_a_rgb.py`:

```python
"""Solución ejercicio 6: HSV → RGB, la inversa de `applied_cv.image.rgb_to_hsv`.

El tono se parte en 6 sectores de 60°; en cada uno un canal vale V, otro p = V(1 − S) y el tercero
sube (t) o baja (q) linealmente con la fracción f del sector.
"""

import numpy as np


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hsv = np.asarray(hsv, dtype=float)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h6 = (h % 1.0) * 6
    sector = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    table = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    channels = [np.choose(sector, [row[c] for row in table]) for c in range(3)]
    return np.stack(channels, axis=-1)


def check() -> None:
    from applied_cv.image import rgb_to_hsv
    from skimage.color import hsv2rgb

    rng = np.random.default_rng(0)
    rgb = rng.random((30, 30, 3))
    np.testing.assert_allclose(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, atol=1e-12)
    hsv = rng.random((30, 30, 3))
    np.testing.assert_allclose(hsv_to_rgb(hsv), hsv2rgb(hsv), atol=1e-12)
    print("ej06 ok — ida y vuelta exacta y coincide con skimage.color.hsv2rgb")


if __name__ == "__main__":
    check()
```

- [ ] **Step 3: Enunciados**

Para cada solución, crear `M/ejercicios/ejNN_<nombre>.py` con: el mismo docstring de la primera línea
cambiando «Solución ejercicio N» por «Ejercicio N» y agregando «Ver README.md.», los mismos imports,
la firma y el docstring de la función con cuerpo `raise NotImplementedError`, y el mismo `check()` y
`if __name__ == "__main__"`. Ejemplo completo para el 1:

```python
"""Ejercicio 1: redimensionar con interpolación bilineal. Ver README.md."""

import numpy as np


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Imagen gris [H, W] → [out_h, out_w], como F.interpolate(mode="bilinear", align_corners=False)."""
    raise NotImplementedError


def check() -> None:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    for h, w, oh, ow in [(5, 7, 11, 13), (8, 8, 3, 5), (6, 4, 6, 4), (4, 9, 10, 2)]:
        x = rng.random((h, w))
        ref = F.interpolate(
            torch.tensor(x)[None, None], size=(oh, ow), mode="bilinear", align_corners=False
        )[0, 0].numpy()
        np.testing.assert_allclose(resize_bilinear(x, oh, ow), ref, atol=1e-12)
    print("ej01 ok — coincide con torch.nn.functional.interpolate")


if __name__ == "__main__":
    check()
```

Los docstrings de función de los enunciados 2–6:
- `iou`: «IoU de dos cajas (x1, y1, x2, y2); 0 si no se intersectan o si la unión no tiene área.»
- `soft_nms`: «Soft-NMS gaussiano: (índices conservados en orden, puntajes finales).»
- `conv_output_size`: «Tamaño de salida de una convolución.»; `receptive_field`: «Campo receptivo
  de una pila de capas [(kernel, stride, dilation), ...].»
- `otsu_threshold`: «Umbral de Otsu de una imagen gris entera: los pixeles > t son primer plano.»
- `hsv_to_rgb`: «HSV en [0, 1] → RGB en [0, 1].»

`M/ejercicios/README.md`:

```markdown
# Ejercicios · Visión por computador

Cada archivo tiene un `check()` que compara tu implementación con una referencia (torch,
torchvision o scikit-image). Ejecuta `uv run python ejercicios/ej01_bilineal.py` desde la carpeta del
módulo. Las soluciones comentadas están en `soluciones/`.

| # | Ejercicio | Qué practica |
|---|---|---|
| 1 | `ej01_bilineal.py` | Interpolación bilineal y la convención `align_corners=False` |
| 2 | `ej02_iou_bordes.py` | IoU sin NaN ni errores en cajas que se tocan o no tienen área |
| 3 | `ej03_soft_nms.py` | Soft-NMS: bajar puntajes en vez de borrar cajas superpuestas |
| 4 | `ej04_campo_receptivo.py` | Tamaño de salida y campo receptivo, verificados con gradientes |
| 5 | `ej05_otsu.py` | Umbral de Otsu con sumas acumuladas |
| 6 | `ej06_hsv_a_rgb.py` | La inversa de `rgb_to_hsv` por sectores de tono |
```

- [ ] **Step 4: Verificar**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_soluciones.py -q && for f in 02-semestre-2/01-vision-por-computador/ejercicios/ej0*.py; do uv run python $f 2>&1 | tail -1; done`
Expected: `7 passed`; cada enunciado termina en `NotImplementedError`.

- [ ] **Step 5: Commit**

```bash
git rm -q 02-semestre-2/01-vision-por-computador/ejercicios/soluciones/.gitkeep
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/ejercicios 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): seis ejercicios con soluciones verificadas"
```

---

### Task 5: Notebooks de teoría (smoke)

**Files:**
- Create: `M/notebooks/01-imagenes-color-y-filtros.ipynb`, `02-transfer-learning.ipynb`,
  `03-deteccion.ipynb`, `04-segmentacion.ipynb`, `05-interpretabilidad-y-atajos.ipynb`
  (generados por un script en el scratchpad con nbformat; metadata `"smoke": true`)
- Delete: `M/notebooks/.gitkeep`

**Interfaces:**
- Consumes: `applied_cv.image`, `applied_cv.boxes`, `applied_cv.masks`, `applied_cv.unet.UNet`,
  `applied_cv.gradcam.grad_cam`, `common.set_seed`, `common.setup_style`.

Cada notebook: celdas cortas, `rng = set_seed(42)`, datos sintéticos, sin descargas, < 30 s en CPU.
Las celdas de lectura (markdown) son conceptuales; las cifras las imprime el código. Después de
ejecutar, revisar que cada lectura sea coherente con la salida real y ajustarla si no.

- [ ] **Step 1: Script generador**

Crear `<scratchpad>/make_nbs_vision.py` (recibe la carpeta destino como argumento) con:

```python
"""Genera los notebooks de teoría de Visión por computador (smoke)."""

import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = """import matplotlib.pyplot as plt
import numpy as np

from common import set_seed, setup_style

setup_style()
rng = set_seed(42)"""


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "smoke": True,
    }
    nbf.write(nb, OUT / name)


SMALL_CNN = """import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def train(model, X, y, epochs=8, lr=1e-3, params=None):
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(42)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(X), generator=gen).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(model(X[idx]), y[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return (model(X).argmax(1) == y).float().mean().item()"""

save("01-imagenes-color-y-filtros.ipynb", [
    md("""# 01 · Imágenes, color y filtros

Una imagen es una matriz de números. Antes de las redes, la visión se hacía con operaciones fijas:
convoluciones (suavizado, bordes), espacios de color e histogramas de textura. Siguen siendo útiles
como baseline y explican qué aprende la primera capa de una CNN. Usaremos una «mazorca» sintética:
una elipse verde-amarilla con manchas café."""),
    code(SETUP + """
from applied_cv.image import (
    color_histogram, conv2d, local_binary_pattern, rgb_to_gray, rgb_to_hsv, sobel,
)


def pod(n_spots, size=96):
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = np.full((size, size, 3), [0.15, 0.35, 0.12])  # follaje
    body = ((xx - 0.5) / 0.28) ** 2 + ((yy - 0.5) / 0.42) ** 2 < 1
    img[body] = [0.75, 0.70, 0.20]  # mazorca amarillo-verdosa
    for _ in range(n_spots):
        cy, cx, r = rng.uniform(0.25, 0.75), rng.uniform(0.35, 0.65), rng.uniform(0.04, 0.09)
        img[body & (((xx - cx) ** 2 + (yy - cy) ** 2) < r**2)] = [0.40, 0.22, 0.08]  # café
    return (img + rng.normal(0, 0.03, img.shape)).clip(0, 1)


healthy, sick = pod(0), pod(5)
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
for a, im, t in zip(ax, (healthy, sick), ("sana", "con manchas")):
    a.imshow(im); a.set_title(t); a.axis("off")"""),
    md("""## 1. Convolución: suavizado y bordes

`conv2d` desliza un núcleo y suma productos. Un núcleo de promedios suaviza; los de Sobel responden
a cambios bruscos de intensidad (bordes)."""),
    code("""gray = rgb_to_gray(sick)
blur = conv2d(gray, np.full((5, 5), 1 / 25))
edges = sobel(gray)
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
for a, im, t in zip(ax, (gray, blur, edges), ("gris", "promedio 5×5", "Sobel")):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")"""),
    md("""## 2. Color: RGB frente a HSV

En RGB, «café» y «amarillo oscuro» se mezclan con la iluminación. El **tono** (H) de HSV separa el
color de la luz: las manchas forman un pico propio en el histograma de H."""),
    code("""hsv_h, hsv_s = rgb_to_hsv(healthy), rgb_to_hsv(sick)
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(hsv_h[..., 0].ravel(), bins=48, range=(0, 0.3), alpha=0.6, label="sana")
ax.hist(hsv_s[..., 0].ravel(), bins=48, range=(0, 0.3), alpha=0.6, label="con manchas")
ax.set(xlabel="tono (0 = rojo, 0.17 = amarillo)", ylabel="pixeles"); ax.legend()
diff = np.abs(color_histogram(healthy) - color_histogram(sick)).sum()
print(f"distancia L1 entre histogramas HSV: {diff:.3f}")"""),
    md("""## 3. Textura: LBP

El patrón binario local compara cada pixel con sus 8 vecinos. Una superficie lisa produce pocos
códigos; una con bordes y puntos, muchos. Es independiente del brillo global."""),
    code("""codes_h = local_binary_pattern(np.round(rgb_to_gray(healthy) * 255))
codes_s = local_binary_pattern(np.round(rgb_to_gray(sick) * 255))
for name, c in (("sana", codes_h), ("con manchas", codes_s)):
    p = np.bincount(c.ravel(), minlength=256) / c.size
    entropy = -(p[p > 0] * np.log2(p[p > 0])).sum()
    print(f"{name:12s} códigos distintos: {np.count_nonzero(p):3d}  entropía: {entropy:.2f} bits")"""),
    md("""**Lectura.** Color (histograma de tono) y textura (LBP) separan estas dos imágenes sin
aprender nada. El baseline «color + textura» del proyecto usa exactamente estos rasgos con una
regresión logística: si una CNN no lo supera, no está viendo más que «manchas café»."""),
])

save("02-transfer-learning.ipynb", [
    md("""# 02 · Transfer learning

Una red entrenada en una tarea A aprende filtros (bordes, rayas, texturas) que sirven para otra
tarea B. Con pocos datos en B, reutilizarlos gana a entrenar desde cero. Lo vemos en pequeño: A =
orientación de rayas (muchos datos), B = frecuencia de rayas (30 ejemplos por clase)."""),
    code(SETUP + "\n" + SMALL_CNN + """


def gratings(n, task, size=32):
    yy, xx = np.mgrid[0:size, 0:size] / size
    X, y = np.empty((n, 1, size, size), np.float32), np.empty(n, np.int64)
    for i in range(n):
        if task == "A":  # 4 orientaciones, frecuencia al azar
            label = rng.integers(4)
            angle, freq = label * np.pi / 4, rng.uniform(3, 6)
        else:  # 3 frecuencias, orientación al azar
            label = rng.integers(3)
            angle, freq = rng.uniform(0, np.pi), (2, 4, 7)[label]
        wave = np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)) + rng.uniform(0, 6))
        X[i, 0] = wave + rng.normal(0, 0.8, (size, size))
        y[i] = label
    return torch.tensor(X), torch.tensor(y)


torch.manual_seed(42)
XA, yA = gratings(3000, "A")
XB, yB = gratings(90, "B")
XB_test, yB_test = gratings(600, "B")
print(XA.shape, XB.shape, XB_test.shape)"""),
    code("""base = train(SmallCNN(4), XA, yA, epochs=6)
print(f"tarea A (orientación), exactitud en train: {accuracy(base, XA, yA):.2f}")"""),
    code("""import copy

torch.manual_seed(0)
scratch = train(SmallCNN(3), XB, yB, epochs=40)

probe = copy.deepcopy(base)
probe.head = nn.Linear(64, 3)
for p in probe.features.parameters():
    p.requires_grad_(False)
probe = train(probe, XB, yB, epochs=40, params=probe.head.parameters())

finetuned = copy.deepcopy(base)
finetuned.head = nn.Linear(64, 3)
finetuned = train(finetuned, XB, yB, epochs=40, lr=3e-4)

for name, m in (("desde cero", scratch), ("sonda lineal", probe), ("fine-tuning", finetuned)):
    print(f"{name:13s} exactitud en test B: {accuracy(m, XB_test, yB_test):.2f}")"""),
    md("""**Lectura.** Con 90 ejemplos, la red desde cero tiene que aprender a la vez los filtros y la
decisión; las transferidas solo ajustan la decisión (sonda) o la ajustan y retocan los filtros con
una tasa de aprendizaje baja (fine-tuning). En el proyecto, A es ImageNet (1,2 millones de fotos) y B
son mazorcas: la pregunta es cuánto sirve un preentrenamiento de un dominio tan distinto."""),
])

save("03-deteccion.ipynb", [
    md("""# 03 · Detección: IoU, NMS y AP

Un detector propone cajas con puntaje. Para evaluarlo hay que decidir cuándo una caja «acierta»
(IoU con la caja real), limpiar duplicados (NMS) y resumir el compromiso precisión-recall (AP)."""),
    code(SETUP + """
from matplotlib.patches import Rectangle

from applied_cv.boxes import average_precision, average_precision_range, box_iou, nms

a, b = np.array([10, 10, 50, 90]), np.array([20, 15, 60, 95])
print(f"IoU = {box_iou(a, b)[0, 0]:.3f}")"""),
    md("""## 1. NMS sobre propuestas ruidosas"""),
    code("""gt = np.array([[10, 10, 40, 80], [55, 20, 85, 90]])
props, scores = [], []
for box in gt:
    for _ in range(6):
        props.append(box + rng.normal(0, 4, 4))
        scores.append(rng.uniform(0.5, 1.0))
props, scores = np.array(props), np.array(scores)
keep = nms(props, scores, 0.5)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for a, idx, t in zip(ax, (np.arange(len(props)), keep), ("propuestas", f"tras NMS ({len(keep)})")):
    a.set(xlim=(0, 100), ylim=(100, 0), title=t)
    for x1, y1, x2, y2 in props[idx]:
        a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="tab:orange"))
    for x1, y1, x2, y2 in gt:
        a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="k", lw=2))"""),
    md("""## 2. AP50 frente a AP50:95

AP50 pide IoU ≥ 0,5 (acertar «más o menos»); AP50:95 promedia umbrales de 0,5 a 0,95 y premia cajas
ajustadas. Con más ruido de localización, AP50 casi no cambia y AP50:95 cae."""),
    code("""def synthetic_eval(noise, n_images=60):
    pb, ps, gts = [], [], []
    for _ in range(n_images):
        g = np.array([[20, 20, 50, 90]]) + rng.normal(0, 5, 4)
        pred = np.r_[g + rng.normal(0, noise, (1, 4)), rng.uniform(0, 80, (1, 2)).repeat(2, 1) + [0, 0, 20, 20]]
        pb.append(pred); ps.append(np.array([rng.uniform(0.6, 1), rng.uniform(0, 0.7)])); gts.append(g)
    return average_precision(pb, ps, gts), average_precision_range(pb, ps, gts)


for noise in (1, 4, 8):
    ap50, ap5095 = synthetic_eval(noise)
    print(f"ruido {noise} px → AP50 = {ap50:.2f}   AP50:95 = {ap5095:.2f}")"""),
    md("""## 3. Anclas: por qué importa la forma

Faster R-CNN parte de cajas «ancla» de varias escalas y proporciones. Una mazorca es alta y
angosta: solo las anclas con proporción alto/ancho ≈ 2 se le parecen."""),
    code("""def anchors(stride=16, size=256, scales=(32, 64, 128), ratios=(0.5, 1.0, 2.0)):
    centers = np.arange(stride / 2, size, stride)
    out = []
    for cy in centers:
        for cx in centers:
            for s in scales:
                for r in ratios:  # r = alto / ancho
                    w, h = s / np.sqrt(r), s * np.sqrt(r)
                    out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, r))
    return np.array(out)


A = anchors()
pod_box = np.array([100, 60, 150, 170])  # 50 × 110
ious = box_iou(pod_box, A[:, :4])[0]
for r in (0.5, 1.0, 2.0):
    print(f"proporción {r}: mejor IoU {ious[A[:, 4] == r].max():.2f}, anclas con IoU ≥ 0.5: {(ious[A[:, 4] == r] >= 0.5).sum()}")"""),
    md("""**Lectura.** En el proyecto el detector solo tiene una clase («mazorca»): la pregunta difícil
(la etapa de monilia) queda para el clasificador. Reportaremos AP50 y AP50:95 del detector en test."""),
])

save("04-segmentacion.ipynb", [
    md("""# 04 · Segmentación con una U-Net

Segmentar es clasificar cada pixel. Comparamos un umbral fijo de brillo con una U-Net pequeña sobre
«mazorcas» sintéticas con sombras, y medimos con Dice e IoU."""),
    code(SETUP + """
import torch
from torch import nn

from applied_cv.masks import dice, mask_iou
from applied_cv.unet import UNet


def sample(size=32):
    yy, xx = np.mgrid[0:size, 0:size] / size
    cy, cx = rng.uniform(0.35, 0.65, 2)
    mask = ((xx - cx) / rng.uniform(0.15, 0.25)) ** 2 + ((yy - cy) / rng.uniform(0.25, 0.35)) ** 2 < 1
    light = 0.4 + 0.6 * xx  # iluminación lateral
    img = np.where(mask, 0.7, 0.45) * light + rng.normal(0, 0.06, (size, size))
    return img.astype(np.float32), mask


data = [sample() for _ in range(400)]
X = torch.tensor(np.stack([d[0] for d in data]))[:, None]
Y = torch.tensor(np.stack([d[1] for d in data])).long()
Xtr, Ytr, Xte, Yte = X[:300], Y[:300], X[300:], Y[300:]
print(Xtr.shape, Xte.shape)"""),
    code("""thr = float(Xtr.mean())
base_dice = np.mean([dice(x[0].numpy() > thr, y.numpy()) for x, y in zip(Xte, Yte)])
print(f"umbral fijo de brillo: Dice = {base_dice:.3f}")"""),
    code("""torch.manual_seed(42)
net = UNet(1, 2, base=8)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for step in range(150):
    idx = torch.randint(0, len(Xtr), (32,))
    opt.zero_grad()
    loss = nn.functional.cross_entropy(net(Xtr[idx]), Ytr[idx])
    loss.backward()
    opt.step()
net.eval()
with torch.no_grad():
    pred = net(Xte).argmax(1).numpy()
scores = [(dice(p, y.numpy()), mask_iou(p, y.numpy())) for p, y in zip(pred, Yte)]
print(f"U-Net: Dice = {np.mean([s[0] for s in scores]):.3f}  IoU = {np.mean([s[1] for s in scores]):.3f}")"""),
    code("""fig, ax = plt.subplots(3, 4, figsize=(8, 6))
for j in range(4):
    for i, (im, t) in enumerate(((Xte[j, 0], "imagen"), (Yte[j], "real"), (pred[j], "U-Net"))):
        ax[i, j].imshow(im, cmap="gray"); ax[i, j].axis("off"); ax[i, j].set_title(t, fontsize=8)"""),
    md("""**Lectura.** La iluminación lateral hace que el umbral global confunda el fondo iluminado
con la mazorca a la sombra; la U-Net usa contexto (forma y bordes) y no solo el brillo del pixel.
Dice siempre es ≥ IoU (IoU = Dice / (2 − Dice))."""),
])

save("05-interpretabilidad-y-atajos.ipynb", [
    md("""# 05 · Grad-CAM y aprendizaje por atajos

Un modelo puede acertar por la razón equivocada. Construimos un atajo a propósito: en train, todas
las imágenes de la clase 1 tienen una «marca de agua» en una esquina (como el celular que tomó casi
todas las m1 en el dataset de cacao). Grad-CAM muestra dónde mira la red."""),
    code(SETUP + "\n" + SMALL_CNN + """
from applied_cv.gradcam import grad_cam


def bars(n, watermark, size=32):
    X, y = np.zeros((n, 1, size, size), np.float32), rng.integers(0, 2, n)
    for i in range(n):
        r, c, L = rng.integers(8, 24), rng.integers(8, 24), rng.integers(6, 12)
        if y[i]:
            X[i, 0, r - L // 2 : r + L // 2, c] = 0.6  # barra vertical (débil)
        else:
            X[i, 0, r, c - L // 2 : c + L // 2] = 0.6  # barra horizontal (débil)
        if watermark == "correlated" and y[i] or watermark == "random" and rng.random() < 0.5:
            X[i, 0, 1:4, 1:4] = 1.0
    X += rng.normal(0, 0.35, X.shape).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)


torch.manual_seed(42)
Xtr, ytr = bars(2000, "correlated")
X_same, y_same = bars(500, "correlated")
X_none, y_none = bars(500, "none")
shortcut = train(SmallCNN(2), Xtr, ytr, epochs=6)
print(f"test con la marca: {accuracy(shortcut, X_same, y_same):.2f}")
print(f"test sin la marca: {accuracy(shortcut, X_none, y_none):.2f}")"""),
    code("""cams = grad_cam(shortcut, shortcut.features[-2], X_same[:6], target=1)
fig, ax = plt.subplots(2, 6, figsize=(10, 3.5))
for j in range(6):
    ax[0, j].imshow(X_same[j, 0], cmap="gray"); ax[0, j].set_title(f"clase {int(y_same[j])}", fontsize=8)
    ax[1, j].imshow(cams[j], cmap="inferno")
    for a in ax[:, j]:
        a.axis("off")"""),
    code("""Xr, yr = bars(2000, "random")
honest = train(SmallCNN(2), Xr, yr, epochs=6)
print(f"modelo sin atajo — test sin la marca: {accuracy(honest, X_none, y_none):.2f}")"""),
    md("""**Lectura.** Con la marca correlacionada con la clase, la red la usa (es más fácil que la
barra) y su exactitud se desploma cuando la marca desaparece; Grad-CAM se enciende en la esquina.
Si la marca aparece al azar en ambas clases, deja de ser útil y la red aprende la barra. En el cacao no
podemos borrar el celular de las fotos: por eso el proyecto mide el atajo (H3) y evalúa h0 vs. m1
dentro de un solo celular."""),
])
print("ok")
```

- [ ] **Step 2: Generar, formatear y ejecutar**

Run:
```bash
git rm -q 02-semestre-2/01-vision-por-computador/notebooks/.gitkeep
uv run python <scratchpad>/make_nbs_vision.py 02-semestre-2/01-vision-por-computador/notebooks
uv run ruff format -q && uv run ruff check --fix -q
uv run python scripts/run_notebooks.py --inplace 02-semestre-2/01-vision-por-computador/notebooks/*.ipynb
```
Expected: `ok` y cinco líneas `ok … (N s)` con N < 30. Si ruff marca E501 en un notebook, partir la
línea en el script generador y regenerar.

- [ ] **Step 3: Revisar las salidas**

Imprimir los textos de salida (script como en el integrador S1) y comprobar: en 02, sonda lineal y
fine-tuning > desde cero; en 03, AP50:95 cae con el ruido más que AP50 y la proporción 2.0 tiene el
mejor IoU; en 04, Dice de la U-Net > umbral; en 05, exactitud «sin la marca» del modelo con atajo
claramente menor que «con la marca», y el modelo sin atajo alta. Si alguna no se cumple, ajustar la
semilla de datos o las épocas en el generador (no la lectura) y anotar el ajuste en el ledger.

- [ ] **Step 4: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q && uv run python scripts/run_notebooks.py
git add 02-semestre-2/01-vision-por-computador/notebooks
git commit -m "feat(vision): cinco notebooks de teoría (smoke)"
```

---
### Task 6: Datos del cacao — anotaciones, preparación, descarga y fixture

**Files:**
- Create: `M/src/applied_cv/cocoa/__init__.py`, `M/src/applied_cv/cocoa/data.py`,
  `M/src/applied_cv/cocoa/prepare.py`, `M/tests/conftest.py`
- Modify: `M/data/download.py` (reemplaza la plantilla)
- Test: `M/tests/test_cocoa_data.py`

**Interfaces:**
- Consumes: `applied_cv.boxes.box_iou`.
- Produces:
  - `CLASSES = ("h0", "m1", "m2", "m3")`.
  - `parse_filename(name) -> (camera: str, timestamp: datetime | None, seq: int | None)`; cámaras:
    `"IMG"`, `"fecha_hora"`, `"epoch"`, `"secuencia"` (`IMG_E####`/`IMG_####`), `"otro"`.
  - `load_annotations(root) -> (images, pods)`: `images` con columnas `image, folder, path, width,
    height, camera, timestamp, seq`; `pods` con `pod_id, image, x1, y1, x2, y2, label` (cajas en
    píxeles originales; `label` ∈ h0…m4).
  - `square_crop_box(box, width, height, margin=0.1) -> (left, top, right, bottom)`.
  - `prepare(raw_root, out_dir, max_side=1024, crop_size=224, margin=0.1, apply_exif=False) -> dict`
    con claves `images, pods, missing, size_mismatch, exif_orientations`; escribe
    `out_dir/images/<image>`, `out_dir/crops/<pod_id>.jpg`, `out_dir/images.csv` (más `scale`,
    `exif`), `out_dir/pods.csv` (cajas en píxeles de la imagen preparada, más `crop`).
  - Fixture `cocoa_raw(tmp_path) -> Path` (carpeta con `CocoaMoniliaDataSet/` y las listas) y la
    función `make_cocoa_dataset(root, n_extra=0, seed=0) -> Path` en `M/tests/conftest.py`.

- [ ] **Step 1: Fixture sintético**

`M/tests/conftest.py`:

```python
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
    arr = np.full((h, w, 3), (30, 70, 30), np.uint8) + rng.integers(0, 20, (h, w, 3), dtype=np.uint8)
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
```

- [ ] **Step 2: Tests (fallan)**

`M/tests/test_cocoa_data.py`:

```python
from datetime import datetime

import pandas as pd
import pytest
from applied_cv.cocoa.data import load_annotations, parse_filename
from applied_cv.cocoa.prepare import prepare, square_crop_box
from PIL import Image


@pytest.mark.parametrize(
    ("name", "camera", "timestamp", "seq"),
    [
        ("IMG_20240226_143016382_HDR.jpg", "IMG", datetime(2024, 2, 26, 14, 30, 16), None),
        ("20241011_110458.jpg", "fecha_hora", datetime(2024, 10, 11, 11, 4, 58), None),
        ("1728667861304.jpg", "epoch", datetime(2024, 10, 11, 17, 31, 1, 304000), None),
        ("IMG_E2361.jpg", "secuencia", None, 2361),
        ("IMG_2352.jpg", "secuencia", None, 2352),
        ("m0 (95).jpg", "otro", None, None),
    ],
)
def test_parse_filename_formats(name, camera, timestamp, seq):
    assert parse_filename(name) == (camera, timestamp, seq)


def test_load_annotations_dedupes_photos_and_pods(cocoa_raw):
    images, pods = load_annotations(cocoa_raw / "CocoaMoniliaDataSet")
    assert len(images) == 11 and images.image.is_unique
    shared = images.set_index("image").loc["IMG_20240227_090000000.jpg"]
    assert shared.folder == "h0" and shared.path == "cocoapods_images/h0/IMG_20240227_090000000.jpg"
    assert len(pods) == 12 and pods.pod_id.tolist() == list(range(12))
    assert pods.label.value_counts().to_dict() == {"h0": 4, "m1": 4, "m3": 2, "m2": 1, "m4": 1}
    first = pods[pods.image == "1728667861304.jpg"].iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == (40, 20, 70, 100)


def test_square_crop_box_is_square_and_clipped():
    # caja 30×80 → lado 96 centrado en (55, 60) → (7, 12, 103, 108), dentro de 160×120
    assert square_crop_box((40, 20, 70, 100), 160, 120, margin=0.1) == (7, 12, 103, 108)
    left, top, right, bottom = square_crop_box((0, 12, 16, 36), 48, 64, margin=0.1)
    assert (left, top) == (0, 10) and right <= 48 and bottom <= 64


def test_prepare_resizes_crops_and_reports_missing(cocoa_raw, tmp_path):
    out = tmp_path / "prep"
    summary = prepare(cocoa_raw / "CocoaMoniliaDataSet", out, max_side=64, crop_size=32)
    assert summary["missing"] == ["IMG_20240229_110000000.jpg"]
    assert summary["images"] == 10 and summary["pods"] == 11
    images = pd.read_csv(out / "images.csv", parse_dates=["timestamp"])
    pods = pd.read_csv(out / "pods.csv")
    assert len(images) == 10 and len(pods) == 11
    with Image.open(out / "images" / "1728667861304.jpg") as im:
        assert im.size == (64, 48)
    with Image.open(out / pods.crop.iloc[0]) as im:
        assert im.size == (32, 32)
    first = pods[pods.image == "1728667861304.jpg"].iloc[0]
    assert (first.x1, first.y1, first.x2, first.y2) == pytest.approx((16, 8, 28, 40))


def test_prepare_reports_exif_and_size_mismatch(cocoa_raw, tmp_path):
    summary = prepare(cocoa_raw / "CocoaMoniliaDataSet", tmp_path / "p", max_side=64, crop_size=32)
    assert summary["size_mismatch"] == ["m0 (95).jpg"]
    assert summary["exif_orientations"] == {1: 10}
```

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_data.py -q`
Expected: ERROR `No module named 'applied_cv.cocoa'`.

Nota sobre el epoch: `1728667861304` ms = 2024-10-11 17:31:01.304 UTC; se guarda sin zona horaria.

- [ ] **Step 3: Implementar**

`M/src/applied_cv/cocoa/__init__.py`:

```python
"""Proyecto del módulo: detección temprana de monilia (Moniliophthora roreri) en mazorcas de cacao.

Datos: CocoaMoniliaDataSet v2 (Zenodo 17716661, CC BY 4.0), fotos de dos fincas de Santander y
Antioquia. Requiere el extra `dl`.
"""

from applied_cv.cocoa.data import CLASSES

__all__ = ["CLASSES"]
```

`M/src/applied_cv/cocoa/data.py`:

```python
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
    `dedupe_iou` con otra de la misma foto) se descartan. Cajas COCO [x, y, w, h] → [x1, y1, x2, y2].
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
```

`M/src/applied_cv/cocoa/prepare.py`:

```python
"""Preparación única: fotos reducidas para el detector y recortes cuadrados por mazorca."""

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
            crop = resized.crop(square_crop_box(box, resized.width, resized.height, margin))
            crop_path = f"crops/{pod.pod_id}.jpg"
            crop.resize((crop_size, crop_size), Image.BILINEAR).save(out_dir / crop_path, quality=95)
            kept_pods.append(
                {"pod_id": pod.pod_id, "image": pod.image, "x1": box[0], "y1": box[1],
                 "x2": box[2], "y2": box[3], "label": pod.label, "crop": crop_path}
            )
    pd.DataFrame(kept_images).to_csv(out_dir / "images.csv", index=False)
    pd.DataFrame(kept_pods).to_csv(out_dir / "pods.csv", index=False)
    return {
        "images": len(kept_images),
        "pods": len(kept_pods),
        "missing": missing,
        "size_mismatch": mismatch,
        "exif_orientations": dict(orientations),
    }
```

`M/data/download.py`:

```python
"""Descarga CocoaMoniliaDataSet v2 (Zenodo 17716661, CC BY 4.0) y prepara fotos y recortes.

1 953 fotos de mazorcas de cacao (fincas de Santander y Antioquia) anotadas en cuatro etapas de
monilia: h0 sana, m1 jorobas, m2 manchas café, m3 esporulación. Cita: CocoaMoniliaDataSet, Data in
Brief (2025), DOI 10.1016/j.dib.2025.112447.

El zip pesa 6,2 GB: la descarga se reanuda si se corta y se verifica con el MD5 de Zenodo (y con el
SHA-256 fijado abajo). Solo se extraen las anotaciones COCO y las fotos.

Uso: uv run python 02-semestre-2/01-vision-por-computador/data/download.py [--borrar-zip]
"""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from applied_cv.cocoa.prepare import prepare

DATA_DIR = Path(__file__).resolve().parent
URL = "https://zenodo.org/records/17716661/files/{name}?download=1"
MD5 = {
    "CocoaMoniliaDataSet.zip": "d3531c4a62f3aaff9c146c7a8ae39a6b",
    "train.txt": "9b68d167efd0b55f6cb772b73eba0698",
    "val.txt": "b1ff099b8ec652a669d10deb0d9758eb",
    "test.txt": "6d9b3e6e3c7dbdbe01a86d18575f56db",
}
SHA256_ZIP = None  # se fija tras la primera descarga verificada por MD5
KEEP = ("CocoaMoniliaDataSet/COCO_annotations/", "CocoaMoniliaDataSet/cocoapods_images/")


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, dest: Path) -> Path:
    """Descarga con reanudación (cabecera Range) y verifica el MD5 publicado por Zenodo."""
    path = dest / name
    if path.exists() and _digest(path, "md5") == MD5[name]:
        return path
    done = path.stat().st_size if path.exists() else 0
    req = urllib.request.Request(URL.format(name=name), headers={"Range": f"bytes={done}-"})
    with urllib.request.urlopen(req) as r, path.open("ab" if done else "wb") as f:
        if done and r.status != 206:  # el servidor no reanudó: empezar de cero
            f.truncate(0)
        shutil.copyfileobj(r, f, length=1 << 22)
    if _digest(path, "md5") != MD5[name]:
        raise RuntimeError(f"MD5 de {name} no coincide; borra el archivo y vuelve a intentar")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--borrar-zip", action="store_true", help="borrar el zip al terminar")
    args = parser.parse_args()
    for name in MD5:
        print(f"descargando {name}…", flush=True)
        download(name, DATA_DIR)
    zip_path = DATA_DIR / "CocoaMoniliaDataSet.zip"
    sha = _digest(zip_path, "sha256")
    if SHA256_ZIP is not None and sha != SHA256_ZIP:
        raise RuntimeError("SHA-256 del zip no coincide con el fijado")
    print(f"SHA-256 del zip: {sha}")
    raw = DATA_DIR / "raw"
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith(KEEP) and not m.endswith("/")]
        z.extractall(raw, members)
    summary = prepare(raw / "CocoaMoniliaDataSet", DATA_DIR / "prepared")
    print({k: (v if not isinstance(v, list) else f"{len(v)}: {v[:5]}") for k, v in summary.items()})
    if args.borrar_zip:
        zip_path.unlink()
        print("zip borrado (6,2 GB liberados)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_data.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests 02-semestre-2/01-vision-por-computador/data/download.py
git commit -m "feat(vision): anotaciones del cacao, preparación de fotos y recortes, descarga verificada"
```

---

### Task 7: Ráfagas y particiones

**Files:**
- Create: `M/src/applied_cv/cocoa/splits.py`
- Test: `M/tests/test_cocoa_splits.py`

**Interfaces:**
- Consumes: tabla `images` de `prepare` (columnas `image, folder, camera, timestamp, seq`).
- Produces:
  - `assign_bursts(images, gap_seconds=120, seq_gap=3) -> pd.Series[int]` (índice de `images`).
  - `burst_split(images, bursts, seed=42) -> pd.Series[str]` con valores `train`/`val`/`test`
    (≈ 70/10/20, agrupada por ráfaga, estratificada por `folder`).
  - `read_split_lists(directory) -> dict[str, list[str]]` (nombres de archivo por parte).
  - `published_split(images, lists) -> tuple[pd.Series[str], list[str]]`: parte de cada foto
    (`train`/`val`/`test`/`fuera`) y las fotos en conflicto (en dos partes), que quedan `fuera`.

- [ ] **Step 1: Tests (fallan)**

`M/tests/test_cocoa_splits.py`:

```python
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
    images = pd.DataFrame({"image": ["IMG_20240227_090000000.jpg", "1728667861304.jpg", "nuevo.jpg"]})
    part, conflicts = published_split(images, lists)
    assert conflicts == ["IMG_20240227_090000000.jpg"]
    assert part.tolist() == ["fuera", "train", "fuera"]
```

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_splits.py -q`
Expected: ERROR `No module named 'applied_cv.cocoa.splits'`.

- [ ] **Step 2: Implementar**

`M/src/applied_cv/cocoa/splits.py`:

```python
"""Particiones honestas: fotos de la misma ráfaga (misma mazorca o árbol) nunca en dos partes."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Intervalos entre fotos consecutivas del mismo celular en el dataset: mediana 4 s, 90 % ≤ 72 s,
# 95 % ≤ 134 s. Con 120 s una ráfaga es «la misma sesión frente a la misma mazorca o árbol».
GAP_SECONDS = 120


def assign_bursts(
    images: pd.DataFrame, gap_seconds: float = GAP_SECONDS, seq_gap: int = 3
) -> pd.Series:
    """Id de ráfaga por foto.

    Con hora: fotos del mismo celular separadas por ≤ `gap_seconds` de la anterior. Sin hora pero
    numeradas (IMG_E####): números a ≤ `seq_gap` del anterior. Sin ninguna de las dos: cada foto sola.
    """
    burst = pd.Series(-1, index=images.index, dtype=int)
    next_id = 0
    timed = images[images["timestamp"].notna()].sort_values(["camera", "timestamp"])
    prev_camera, prev_time = None, None
    for idx, row in timed.iterrows():
        if row.camera != prev_camera or (row.timestamp - prev_time).total_seconds() > gap_seconds:
            next_id += 1
        burst[idx] = next_id
        prev_camera, prev_time = row.camera, row.timestamp
    numbered = images[images["timestamp"].isna() & images["seq"].notna()].sort_values("seq")
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
    return {
        part: [Path(line.strip()).name for line in (directory / f"{part}.txt").read_text().split()]
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
```

- [ ] **Step 3: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_splits.py -q`
Expected: `3 passed`. (Las listas del fixture tienen una ruta por entrada de los JSON: 4 + 5 + 1 + 2
= 12, contando la foto repetida en dos carpetas y la que no tiene archivo.)

- [ ] **Step 4: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): ráfagas de fotos y particiones por ráfaga y publicada"
```

---

### Task 8: Rasgos de color y textura, y clasificadores

**Files:**
- Create: `M/src/applied_cv/cocoa/features.py`, `M/src/applied_cv/cocoa/classifiers.py`
- Test: `M/tests/test_cocoa_classifiers.py`

**Interfaces:**
- Consumes: `applied_cv.image.color_histogram`, `lbp_histogram`, `rgb_to_gray`;
  `applied_dl.training.fit` (bucle con early stopping del módulo de aprendizaje profundo; devuelve
  `History` y deja el modelo con los pesos de menor pérdida de validación).
- Produces:
  - `color_texture_features(rgb_uint8: ndarray[H, W, 3]) -> ndarray[304]` (48 de color + 256 de LBP).
  - `IMAGENET_MEAN`, `IMAGENET_STD`; `eval_transform()`; `train_augmentation()` (TODO del usuario,
    versión mínima funcional).
  - `CropDataset(items: list[str | Path | PIL.Image.Image], labels: list[int] | None, transform)`.
  - `build_resnet(n_classes: int, pretrained: bool) -> nn.Module` (ResNet-18; `fc` nueva).
  - `fine_tune(model, train_ds, val_ds, *, epochs, lr, device, batch_size=32) -> History`.
  - `predict_proba(model, ds, device, batch_size=64) -> ndarray[N, n_classes]`.
  - `extract_features(model, ds, device, batch_size=64) -> ndarray[N, 512]` (salida antes de `fc`).
  - `fit_logistic(X_tr, y_tr, X_va, y_va, n_classes=4, Cs=(0.01, 0.1, 1.0, 10.0)) -> LogisticModel`
    con `.predict_proba(X) -> ndarray[N, n_classes]` (columnas completas aunque falte una clase en
    train) y `.C`.

- [ ] **Step 1: Tests (fallan)**

`M/tests/test_cocoa_classifiers.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.classifiers import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    CropDataset,
    build_resnet,
    eval_transform,
    extract_features,
    fine_tune,
    fit_logistic,
    predict_proba,
    train_augmentation,
)
from applied_cv.cocoa.features import color_texture_features  # noqa: E402
from applied_cv.image import rgb_to_hsv  # noqa: E402
from PIL import Image  # noqa: E402

RNG = np.random.default_rng(0)
GREEN, BROWN = (90, 160, 60), (110, 70, 30)


def _crop(color, size=32):
    arr = np.clip(np.array(color) + RNG.integers(-15, 15, (size, size, 3)), 0, 255)
    return arr.astype(np.uint8)


def test_color_texture_features_shape_and_separation():
    crops = [_crop(GREEN) for _ in range(20)] + [_crop(BROWN) for _ in range(20)]
    X = np.stack([color_texture_features(c) for c in crops])
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    assert X.shape == (40, 304)
    model = fit_logistic(X[::2], y[::2], X[1::2], y[1::2], n_classes=4)
    proba = model.predict_proba(X[1::2])
    assert proba.shape == (20, 4) and np.allclose(proba.sum(axis=1), 1)
    assert (proba.argmax(axis=1) == y[1::2]).mean() == 1.0


def test_train_augmentation_preserves_hue():
    # Regla de dominio: el café de la lesión es el síntoma; la aumentación no puede cambiar el tono.
    torch.manual_seed(0)
    image = Image.fromarray(np.full((64, 64, 3), BROWN, np.uint8))
    hue = rgb_to_hsv(np.array(BROWN) / 255)[0]
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    for _ in range(20):
        x = (train_augmentation()(image) * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        hsv = rgb_to_hsv(x)
        visible = hsv[..., 2] > 0.05  # ignora el relleno negro de rotaciones
        assert np.abs(hsv[..., 0][visible] - hue).max() < 0.02


def test_resnet_fine_tune_predict_and_features():
    torch.manual_seed(0)
    images = [Image.fromarray(_crop(c, 64)) for c in [GREEN] * 8 + [BROWN] * 8]
    labels = [0] * 8 + [1] * 8
    train_ds = CropDataset(images, labels, eval_transform())
    model = build_resnet(4, pretrained=False)
    history = fine_tune(
        model, train_ds, train_ds, epochs=2, lr=1e-3, device=torch.device("cpu"), batch_size=8
    )
    assert len(history.train_loss) == 2
    proba = predict_proba(model, CropDataset(images, None, eval_transform()), torch.device("cpu"))
    assert proba.shape == (16, 4) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    feats = extract_features(model, CropDataset(images, None, eval_transform()), torch.device("cpu"))
    assert feats.shape == (16, 512)
    assert isinstance(model.fc, torch.nn.Linear)  # extract_features no rompe el modelo
```

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_classifiers.py -q`
Expected: ERROR `No module named 'applied_cv.cocoa.classifiers'`.

- [ ] **Step 2: Implementar**

`M/src/applied_cv/cocoa/features.py`:

```python
"""Rasgos hechos a mano para el baseline «color + textura»."""

import numpy as np

from applied_cv.image import color_histogram, lbp_histogram, rgb_to_gray


def color_texture_features(rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    """Histogramas HSV (3·bins) + histograma LBP (256) de un recorte RGB uint8."""
    x = np.asarray(rgb, dtype=float) / 255.0
    return np.concatenate([color_histogram(x, bins), lbp_histogram(np.round(rgb_to_gray(x) * 255))])
```

`M/src/applied_cv/cocoa/classifiers.py`:

```python
"""Clasificadores de etapa sobre recortes: logística sobre rasgos y ResNet-18 (sonda y fine-tuning)."""

from pathlib import Path

import numpy as np
import torch
from applied_dl.training import History, fit
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def train_augmentation() -> transforms.Compose:
    """Aumentación del clasificador. Restricción de dominio: no cambiar el tono (test lo verifica).

    TODO(usuario): esta es la versión mínima (solo volteo). Decide qué transformaciones conservan el
    síntoma: ¿rotaciones (la mazorca puede estar inclinada)? ¿brillo y contraste (sol y sombra en
    campo)? ¿recortes aleatorios (que podrían cortar la joroba de una m1)?
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CropDataset(Dataset):
    """Recortes desde archivo o en memoria (los del detector), con etiqueta opcional."""

    def __init__(self, items: list, labels: list[int] | None, transform):
        self.items, self.labels, self.transform = list(items), labels, transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        if isinstance(item, str | Path):
            with Image.open(item) as im:
                image = im.convert("RGB")
        else:
            image = item.convert("RGB")
        x = self.transform(image)
        return x, (self.labels[i] if self.labels is not None else -1)


def build_resnet(n_classes: int, pretrained: bool) -> nn.Module:
    """ResNet-18 con pesos de ImageNet (o al azar) y una capa final nueva de `n_classes`."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def fine_tune(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 32,
) -> History:
    """Ajusta toda la red con AdamW y early stopping sobre la pérdida de validación."""
    gen = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    return fit(
        model, train_loader, val_loader, optimizer, epochs=epochs, patience=3, device=device
    )


@torch.no_grad()
def predict_proba(
    model: nn.Module, ds: Dataset, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    model.to(device).eval()
    out = [
        torch.softmax(model(x.to(device)), dim=1).cpu()
        for x, _ in DataLoader(ds, batch_size=batch_size)
    ]
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_features(
    model: nn.Module, ds: Dataset, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """Vector de 512 de la penúltima capa (sonda lineal y H3). Restaura `fc` al terminar."""
    head, model.fc = model.fc, nn.Identity()
    try:
        model.to(device).eval()
        feats = [model(x.to(device)).cpu() for x, _ in DataLoader(ds, batch_size=batch_size)]
    finally:
        model.fc = head
    return torch.cat(feats).numpy()


class LogisticModel:
    """Escalado + logística multinomial con columnas para todas las clases."""

    def __init__(self, pipeline, n_classes: int, C: float):
        self.pipeline, self.n_classes, self.C = pipeline, n_classes, C

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), self.n_classes))
        proba[:, self.pipeline.classes_] = self.pipeline.predict_proba(X)
        return proba


def fit_logistic(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    n_classes: int = 4,
    Cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> LogisticModel:
    """Elige C por F1-macro en validación; clases balanceadas (h0 es la más frecuente)."""
    best = None
    for C in Cs:
        pipe = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        ).fit(X_tr, y_tr)
        score = f1_score(y_va, pipe.predict(X_va), average="macro")
        if best is None or score > best[0]:
            best = (score, C, pipe)
    return LogisticModel(best[2], n_classes, best[1])
```

`LogisticModel.pipeline.classes_`: `make_pipeline` expone `classes_` del último paso.

- [ ] **Step 3: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_classifiers.py -q`
Expected: `3 passed` (< 30 s en CPU).

- [ ] **Step 4: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): rasgos color+textura, logística y ResNet-18 (sonda y fine-tuning)"
```

---
### Task 9: Evaluación y estadística

**Files:**
- Create: `M/src/applied_cv/cocoa/evaluation.py`
- Test: `M/tests/test_cocoa_evaluation.py`

**Interfaces:**
- Consumes: `applied_cv.boxes.box_iou`.
- Produces:
  - `sick_score(proba) -> ndarray` (= 1 − P(h0)).
  - `threshold_at_fpr(negative_scores, fpr=0.1) -> float` (marcadas: puntaje > umbral; 0.5 si no hay
    negativas).
  - `recall_above(scores, threshold) -> float` (nan si no hay puntajes).
  - `macro_f1(y_true, proba, n_classes=4) -> float`.
  - `cluster_members(clusters) -> list[ndarray]`; `resample(members, rng) -> ndarray`.
  - `bootstrap_diff(stat, clusters, n_boot=2000, seed=42) -> dict(diff, ci, p_value, n_boot)`;
    `stat(idx) -> float` (nan se descarta).
  - `bootstrap_unpaired(stat_a, clusters_a, stat_b, clusters_b, n_boot=2000, seed=42) -> dict`.
  - `holm(p_values: dict[str, float]) -> dict[str, float]`.
  - `decide(result: dict, p_holm: float, alpha=0.05) -> dict` (+ `p_holm`, `rechaza_h0`,
    `apoya_hipotesis` = rechaza ∧ diff > 0).
  - `match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5) -> ndarray[int]` (por caja real, índice de
    la predicción emparejada o −1).

- [ ] **Step 1: Tests (fallan)**

`M/tests/test_cocoa_evaluation.py`:

```python
import numpy as np
import pytest
from applied_cv.cocoa.evaluation import (
    bootstrap_diff,
    bootstrap_unpaired,
    cluster_members,
    decide,
    holm,
    macro_f1,
    match_boxes,
    recall_above,
    resample,
    sick_score,
    threshold_at_fpr,
)


def test_threshold_at_fpr_controls_false_alarms():
    negatives = np.arange(100) / 100
    thr = threshold_at_fpr(negatives, fpr=0.1)
    assert recall_above(negatives, thr) <= 0.1
    assert recall_above(negatives, thr) == pytest.approx(0.09)
    assert threshold_at_fpr(np.array([]), 0.1) == 0.5
    assert np.isnan(recall_above(np.array([]), 0.5))


def test_sick_score_and_macro_f1():
    proba = np.array([[0.9, 0.1, 0, 0], [0.2, 0.7, 0.1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(sick_score(proba), [0.1, 0.8, 1.0, 1.0])
    assert macro_f1(np.array([0, 1, 2, 3]), proba) == 1.0


def test_resample_keeps_clusters_whole():
    clusters = np.array([0, 0, 0, 1, 1, 2])
    members = cluster_members(clusters)
    idx = resample(members, np.random.default_rng(0))
    counts = np.bincount(clusters[idx], minlength=3)
    assert all(c % s == 0 for c, s in zip(counts, (3, 2, 1), strict=True))


def test_bootstrap_identical_and_clear_difference():
    clusters = np.repeat(np.arange(40), 3)
    same = bootstrap_diff(lambda idx: 0.0 * len(idx), clusters, n_boot=200)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0
    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 0.2, 120), rng.normal(0.0, 0.2, 120)
    res = bootstrap_diff(lambda idx: a[idx].mean() - b[idx].mean(), clusters, n_boot=200)
    assert res["ci"][0] > 0 and res["p_value"] == pytest.approx(1 / 200)
    un = bootstrap_unpaired(
        lambda i: a[i].mean(), clusters, lambda i: b[i].mean(), clusters, n_boot=200
    )
    assert un["diff"] == pytest.approx(a.mean() - b.mean()) and un["ci"][0] > 0


def test_holm_and_decide():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    worse = decide({"diff": -0.1, "ci": [-0.2, -0.05], "p_value": 0.001}, p_holm=0.003)
    assert worse["rechaza_h0"] and not worse["apoya_hipotesis"]


def test_match_boxes_greedy_one_to_one():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [50, 50, 60, 60]])
    pred = np.array([[1, 0, 10, 10], [0, 0, 10, 11], [21, 0, 30, 10]])
    np.testing.assert_array_equal(match_boxes(pred, gt), [1, 2, -1])
    np.testing.assert_array_equal(match_boxes(np.zeros((0, 4)), gt), [-1, -1, -1])
```

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_evaluation.py -q`
Expected: ERROR `No module named 'applied_cv.cocoa.evaluation'`.

(En `test_match_boxes…`: IoU(gt0, pred0) = 90/100 = 0.90 y IoU(gt0, pred1) = 100/110 = 0.909, así
que gt0 se queda con pred1; gt1 con pred2; gt2 sin pareja.)

- [ ] **Step 2: Implementar**

`M/src/applied_cv/cocoa/evaluation.py`:

```python
"""Métricas del proyecto y estadística: umbral a FPR fija, bootstrap por ráfaga y Holm."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score

from applied_cv.boxes import box_iou


def sick_score(proba: np.ndarray) -> np.ndarray:
    """P(enferma) = P(m1) + P(m2) + P(m3) = 1 − P(h0)."""
    return 1.0 - np.asarray(proba)[:, 0]


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float = 0.1) -> float:
    """Umbral tal que, en las sanas dadas (validación), la fracción con puntaje > umbral es ≤ fpr.

    Sin sanas (solo pasa con datos de prueba diminutos) devuelve 0.5.
    """
    negative_scores = np.asarray(negative_scores, dtype=float)
    if negative_scores.size == 0:
        return 0.5
    return float(np.quantile(negative_scores, 1 - fpr, method="higher"))


def recall_above(scores: np.ndarray, threshold: float) -> float:
    """Fracción de puntajes > umbral (recall si son enfermas, FPR si son sanas)."""
    scores = np.asarray(scores, dtype=float)
    return float(np.mean(scores > threshold)) if scores.size else float("nan")


def macro_f1(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 4) -> float:
    return float(
        f1_score(
            y_true,
            np.asarray(proba).argmax(axis=1),
            average="macro",
            labels=list(range(n_classes)),
            zero_division=0,
        )
    )


def cluster_members(clusters: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    return [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]


def resample(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Remuestrea conglomerados enteros con reemplazo: las fotos de una ráfaga no son independientes."""
    chosen = rng.integers(0, len(members), len(members))
    return np.concatenate([members[k] for k in chosen])


def _summary(observed: float, dist: np.ndarray) -> dict:
    dist = dist[~np.isnan(dist)]
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p = 2 * min(np.mean(dist <= 0), np.mean(dist >= 0))
    return {
        "diff": float(observed),
        "ci": [float(lo), float(hi)],
        "p_value": float(min(1.0, max(p, 1 / len(dist)))),  # resolución 1/n_boot
        "n_boot": int(len(dist)),
    }


def bootstrap_diff(
    stat: Callable[[np.ndarray], float], clusters: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    """IC y p-valor bilateral de un estadístico de diferencia, remuestreando conglomerados."""
    members, rng = cluster_members(clusters), np.random.default_rng(seed)
    observed = stat(np.arange(len(clusters)))
    return _summary(observed, np.array([stat(resample(members, rng)) for _ in range(n_boot)]))


def bootstrap_unpaired(
    stat_a: Callable[[np.ndarray], float],
    clusters_a: np.ndarray,
    stat_b: Callable[[np.ndarray], float],
    clusters_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Diferencia stat_a − stat_b entre dos conjuntos distintos, remuestreados por separado."""
    rng = np.random.default_rng(seed)
    ma, mb = cluster_members(clusters_a), cluster_members(clusters_b)
    observed = stat_a(np.arange(len(clusters_a))) - stat_b(np.arange(len(clusters_b)))
    dist = np.array([stat_a(resample(ma, rng)) - stat_b(resample(mb, rng)) for _ in range(n_boot)])
    return _summary(observed, dist)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Corrección de Holm-Bonferroni (controla el error familiar sobre todas las hipótesis)."""
    names = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for k, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - k) * p_values[name]))
        adjusted[name] = running
    return adjusted


def decide(result: dict, p_holm: float, alpha: float = 0.05) -> dict:
    """Rechazar H0 no basta: la hipótesis se apoya solo si además la diferencia es positiva."""
    rejects = p_holm < alpha
    return {
        **result,
        "p_holm": float(p_holm),
        "rechaza_h0": bool(rejects),
        "apoya_hipotesis": bool(rejects and result["diff"] > 0),
    }


def match_boxes(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    """Para cada caja real, la predicción emparejada (−1 si ninguna); uno a uno, por IoU descendente."""
    pred = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    match = np.full(len(gt), -1)
    if len(pred) == 0 or len(gt) == 0:
        return match
    ious = box_iou(gt, pred)
    pairs = sorted(
        ((ious[i, j], i, j) for i, j in zip(*np.nonzero(ious >= iou_threshold), strict=True)),
        reverse=True,
    )
    used = set()
    for _, i, j in pairs:
        if match[i] == -1 and j not in used:
            match[i] = j
            used.add(j)
    return match
```

- [ ] **Step 3: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_evaluation.py -q`
Expected: `6 passed`.

- [ ] **Step 4: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): métricas a FPR fija, bootstrap por ráfaga, Holm y emparejamiento de cajas"
```

---

### Task 10: Detector de mazorcas

**Files:**
- Create: `M/src/applied_cv/cocoa/detector.py`
- Test: `M/tests/test_cocoa_detector.py`

**Interfaces:**
- Consumes: imágenes y `pods.csv` de `prepare` (cajas en píxeles de la imagen preparada).
- Produces:
  - `build_detector(pretrained: bool, min_size=512, max_size=640) -> nn.Module` (2 clases: fondo y
    mazorca).
  - `DetectionDataset(paths, boxes, flip_prob=0.0, seed=42)`; ítem = `(Tensor[3, H, W], {"boxes",
    "labels"})`.
  - `collate(batch) -> tuple`.
  - `train_detector(model, ds, *, epochs, device, lr=0.005, batch_size=4, seed=42) -> list[float]`
    (pérdida media por época).
  - `detect(model, paths, device, score_threshold=0.05) -> list[tuple[ndarray[K, 4], ndarray[K]]]`.

- [ ] **Step 1: Tests (fallan)**

`M/tests/test_cocoa_detector.py`:

```python
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.detector import (  # noqa: E402
    DetectionDataset,
    build_detector,
    detect,
    train_detector,
)
from applied_cv.cocoa.prepare import prepare  # noqa: E402


def test_flip_mirrors_boxes(cocoa_raw, tmp_path):
    out = tmp_path / "p"
    prepare(cocoa_raw / "CocoaMoniliaDataSet", out, max_side=64, crop_size=32)
    path = out / "images" / "1728667861304.jpg"
    box = np.array([[16.0, 8.0, 28.0, 40.0]])
    plain = DetectionDataset([path], [box])[0]
    flipped = DetectionDataset([path], [box], flip_prob=1.0)[0]
    width = plain[0].shape[-1]
    torch.testing.assert_close(flipped[0], plain[0].flip(-1))
    torch.testing.assert_close(flipped[1]["boxes"], torch.tensor([[width - 28.0, 8, width - 16, 40]]))
    assert flipped[1]["labels"].tolist() == [1]


def test_train_and_detect_on_tiny_images(cocoa_raw, tmp_path):
    out = tmp_path / "p"
    prepare(cocoa_raw / "CocoaMoniliaDataSet", out, max_side=64, crop_size=32)
    pods = pd.read_csv(out / "pods.csv")
    names = sorted(pods.image.unique())[:4]
    paths = [out / "images" / n for n in names]
    boxes = [pods[pods.image == n][["x1", "y1", "x2", "y2"]].to_numpy() for n in names]
    model = build_detector(pretrained=False, min_size=64, max_size=96)
    losses = train_detector(
        model, DetectionDataset(paths, boxes, flip_prob=0.5), epochs=1, device=torch.device("cpu"),
        batch_size=2,
    )
    assert len(losses) == 1 and np.isfinite(losses[0])
    results = detect(model, paths, torch.device("cpu"), score_threshold=0.0)
    assert len(results) == 4
    assert all(b.shape[1:] == (4,) and len(b) == len(s) for b, s in results)
```

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_detector.py -q`
Expected: ERROR `No module named 'applied_cv.cocoa.detector'`.

- [ ] **Step 2: Implementar**

`M/src/applied_cv/cocoa/detector.py`:

```python
"""Detector de mazorcas (una sola clase): Faster R-CNN MobileNetV3-FPN de torchvision.

Separar «encontrar la mazorca» de «diagnosticar la etapa» permite ver dónde falla el sistema: el
detector solo responde «¿hay una mazorca aquí?»; la etapa la decide el clasificador sobre el recorte.
"""

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor


def build_detector(pretrained: bool, min_size: int = 512, max_size: int = 640) -> nn.Module:
    """Pesos de COCO (o al azar) y una cabeza nueva de 2 clases: fondo y mazorca.

    `min_size`/`max_size`: el modelo reescala cada foto internamente; 512–640 px entrenan en ~2–3 min
    por época en un M4 (MPS).
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


class DetectionDataset(Dataset):
    """Fotos con sus cajas; con `flip_prob` > 0 voltea horizontalmente foto y cajas a la vez."""

    def __init__(self, paths: list, boxes: list[np.ndarray], flip_prob: float = 0.0, seed: int = 42):
        self.paths, self.boxes, self.flip_prob = list(paths), list(boxes), flip_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            image = to_tensor(im.convert("RGB"))
        boxes = torch.as_tensor(np.asarray(self.boxes[i], dtype=np.float32).reshape(-1, 4))
        if self.flip_prob and self.rng.random() < self.flip_prob:
            width = image.shape[-1]
            image = image.flip(-1)
            boxes = torch.stack(
                [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1
            )
        return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch, strict=True))


def train_detector(
    model: nn.Module,
    ds: Dataset,
    *,
    epochs: int,
    device: torch.device,
    lr: float = 0.005,
    batch_size: int = 4,
    seed: int = 42,
) -> list[float]:
    """SGD con momentum y tasa coseno; devuelve la pérdida total media de cada época."""
    torch.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    model.to(device).train()
    history = []
    for _ in range(epochs):
        total = 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        history.append(total / len(loader))
    return history


@torch.no_grad()
def detect(
    model: nn.Module, paths: list, device: torch.device, score_threshold: float = 0.05
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(cajas, puntajes) por foto, ya con NMS (lo aplica el propio Faster R-CNN)."""
    model.to(device).eval()
    results = []
    for path in paths:
        with Image.open(path) as im:
            x = to_tensor(im.convert("RGB")).to(device)
        pred = model([x])[0]
        keep = pred["scores"] >= score_threshold
        results.append((pred["boxes"][keep].cpu().numpy(), pred["scores"][keep].cpu().numpy()))
    return results
```

- [ ] **Step 3: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_cocoa_detector.py -q`
Expected: `2 passed` (< 60 s en CPU).

- [ ] **Step 4: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/src 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): detector de mazorcas Faster R-CNN MobileNetV3"
```

---
### Task 11: Experimentos (`run.py`) y demo (`diagnosticar.py`)

**Files:**
- Create: `M/proyecto/run.py`, `M/proyecto/diagnosticar.py`
- Test: `M/tests/test_run_smoke.py`, `M/tests/test_diagnosticar.py`

**Interfaces:**
- Consumes: todo `applied_cv.cocoa`, `applied_cv.gradcam.grad_cam`, `applied_dl.training.get_device`,
  `common`.
- Produces:
  - `run(data: Path, out: Path, *, smoke=False, device=None, models_dir=None) -> dict`; `data`
    contiene `prepared/` y `train.txt`/`val.txt`/`test.txt`; escribe `out/metrics.json`, figuras, y
    en `models_dir` (por defecto `data/modelos/`) `clasificador.pt`, `detector.pt`, `modelos.json`.
  - `diagnosticar.main(argv, *, models_dir=None) -> int` (0 ok o sin mazorcas, 1 faltan modelos,
    2 archivo inválido).

- [ ] **Step 1: Tests (fallan)**

`M/tests/test_run_smoke.py`:

```python
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.prepare import prepare  # noqa: E402

RUN = Path(__file__).parents[1] / "proyecto" / "run.py"


def _load_run():
    spec = importlib.util.spec_from_file_location("vision_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_smoke_counts_m4(cocoa_raw_large, tmp_path):
    data = tmp_path / "data"
    prepare(cocoa_raw_large / "CocoaMoniliaDataSet", data / "prepared", max_side=160, crop_size=64)
    for part in ("train", "val", "test"):
        shutil.copy(cocoa_raw_large / f"{part}.txt", data / f"{part}.txt")
    report = _load_run().run(
        data, tmp_path / "out", smoke=True, device=torch.device("cpu"), models_dir=tmp_path / "m"
    )
    assert report["datos"]["m4_excluidas"] == 1
    assert report["datos"]["conflictos_publicada"] == ["IMG_20240227_090000000.jpg"]
    assert set(report["clasificacion_rafaga"]) == {
        "prevalencia", "color_textura", "sonda_lineal", "fine_tuning"
    }
    assert set(report["hipotesis"]) == {"H1", "H2", "H3"}
    for h in report["hipotesis"].values():
        assert {"diff", "ci", "p_value", "p_holm", "rechaza_h0", "apoya_hipotesis"} <= set(h)
        assert h["apoya_hipotesis"] == (h["rechaza_h0"] and h["diff"] > 0)
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert saved["datos"] == report["datos"]
    for name in ("clasificador.pt", "detector.pt", "modelos.json"):
        assert (tmp_path / "m" / name).exists()
```

`M/tests/test_diagnosticar.py`:

```python
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from applied_cv.cocoa.classifiers import build_resnet  # noqa: E402
from applied_cv.cocoa.detector import build_detector  # noqa: E402
from PIL import Image  # noqa: E402

DEMO = Path(__file__).parents[1] / "proyecto" / "diagnosticar.py"


def _demo():
    spec = importlib.util.spec_from_file_location("vision_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "modelos"
    d.mkdir()
    torch.save(build_resnet(4, pretrained=False).state_dict(), d / "clasificador.pt")
    torch.save(build_detector(False, 64, 96).state_dict(), d / "detector.pt")
    (d / "modelos.json").write_text(json.dumps({"umbral": 0.5, "det_min_size": 64, "det_max_size": 96}))
    return d


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "foto.jpg"
    Image.fromarray(np.full((96, 128, 3), (90, 160, 60), np.uint8)).save(path)
    return path


def test_missing_file(models_dir, tmp_path, capsys):
    assert _demo().main([str(tmp_path / "no.jpg")], models_dir=models_dir) == 2
    assert "No existe" in capsys.readouterr().out


def test_not_an_image(models_dir, tmp_path, capsys):
    text = tmp_path / "nota.txt"
    text.write_text("hola")
    assert _demo().main([str(text)], models_dir=models_dir) == 2
    assert "no es una imagen" in capsys.readouterr().out


def test_missing_models(photo, tmp_path, capsys):
    assert _demo().main([str(photo)], models_dir=tmp_path / "vacio") == 1
    assert "run.py" in capsys.readouterr().out


def test_no_pods_gives_no_diagnosis(models_dir, photo, capsys):
    assert _demo().main([str(photo), "--min-score", "1.01"], models_dir=models_dir) == 0
    assert "No se detectaron mazorcas" in capsys.readouterr().out


def test_diagnosis_writes_figure(models_dir, photo, tmp_path, capsys):
    out = tmp_path / "salida.png"
    code = _demo().main([str(photo), "--min-score", "0", "--salida", str(out)], models_dir=models_dir)
    assert code == 0
    assert out.exists() or "No se detectaron mazorcas" in capsys.readouterr().out
```

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_run_smoke.py 02-semestre-2/01-vision-por-computador/tests/test_diagnosticar.py -q`
Expected: FAIL (`FileNotFoundError` al cargar `run.py`/`diagnosticar.py`).

- [ ] **Step 2: `run.py`**

`M/proyecto/run.py`:

```python
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
        return {"diff": nan, "ci": [nan, nan], "p_value": 1.0, "n_boot": 0,
                "exactitud_balanceada": nan, "n": counts, "nota": "no evaluable"}
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
    """Detector entrenado en train (por ráfaga); AP en test; sistema completo sobre las fotos de test."""
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
    s_ft, s_ct = sick_score(proba["fine_tuning"]["test"]), sick_score(proba["color_textura"]["test"])
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
        stage[usable].reset_index(drop=True), X[usable], F[usable],
        [p for p, u in zip(paths, usable, strict=True) if u], pub[usable], cfg, device,
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
        lambda i: macro_f1(yt_pub[i], p_pub[i]), bursts_pub,
        lambda i: macro_f1(yt[i], p_raf[i]), bursts_test,
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
    report["sonda_celular"] = {"exactitud_balanceada": h3.pop("exactitud_balanceada"), "n": h3.pop("n")}

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
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="predicha", ylabel="real", title="Fine-tuning · test por ráfaga")
    save_fig(fig, out / "confusion.png")
    plt.close(fig)
    names = list(report["clasificacion_rafaga"])
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(names, [report["clasificacion_rafaga"][n]["recall_m1"] for n in names])
    ax.set(ylabel="recall de m1 (FPR h0 = 10 % en val)", ylim=(0, 1))
    save_fig(fig, out / "recall_m1.png")
    plt.close(fig)
    test_paths = [p for p, k in zip(paths, part, strict=True) if k == "test"]
    gradcam_mosaic(clf, test, yt, proba["fine_tuning"]["test"], thr_ft, test_paths, device,
                   out / "errores_gradcam.png")
    torch.save(clf.cpu().state_dict(), models_dir / "clasificador.pt")
    torch.save(detector.cpu().state_dict(), models_dir / "detector.pt")
    (models_dir / "modelos.json").write_text(json.dumps({
        "umbral": thr_ft, "det_min_size": cfg["det_min_size"], "det_max_size": cfg["det_max_size"],
    }))
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
        print(f"{name}: Δ = {h['diff']:+.4f} IC [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}] "
              f"p_Holm = {h['p_holm']:.4f} → apoya: {h['apoya_hipotesis']}")
```

- [ ] **Step 3: `diagnosticar.py`**

`M/proyecto/diagnosticar.py`:

```python
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
from applied_cv.cocoa.prepare import square_crop_box
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
    try:
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
    except UnidentifiedImageError:
        print(f"{path} no es una imagen que se pueda abrir.")
        return 2
    try:
        detector, classifier, threshold = load_models(models_dir or data_dir(MODULE) / "modelos")
    except FileNotFoundError:
        print("Faltan los modelos: ejecuta primero proyecto/run.py.")
        return 1
    scale = min(1.0, 1024 / max(image.size))
    image = image.resize((round(image.width * scale), round(image.height * scale)))
    device = get_device()
    with torch.no_grad():
        pred = detector.to(device)([to_tensor(image).to(device)])[0]
    boxes = pred["boxes"][pred["scores"] >= args.min_score].cpu().numpy()
    if len(boxes) == 0:
        print("No se detectaron mazorcas en la foto: no hay diagnóstico.")
        return 0
    crops = [image.crop(square_crop_box(b, image.width, image.height)).resize((224, 224)) for b in boxes]
    ds = CropDataset(crops, None, eval_transform())
    proba = predict_proba(classifier, ds, device)
    sick = 1 - proba[:, 0]
    fig, axes = plt.subplots(1, 1 + min(4, len(crops)), figsize=(4 + 2.5 * min(4, len(crops)), 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    for i, (box, p, s) in enumerate(zip(boxes, proba, sick, strict=True)):
        stage = CLASSES[int(p.argmax())]
        alarm = s > threshold
        print(f"mazorca {i + 1}: {STAGE_NAMES[stage]} · P(enferma) = {s:.2f}"
              + (" · ALERTA" if alarm else ""))
        x1, y1, x2, y2 = box
        axes[0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2,
                                    color="tab:red" if alarm else "tab:green"))
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
```

- [ ] **Step 4: Ver que pasan**

Run: `uv run pytest 02-semestre-2/01-vision-por-computador/tests/test_run_smoke.py 02-semestre-2/01-vision-por-computador/tests/test_diagnosticar.py -q`
Expected: `6 passed` (smoke < 2 min en CPU). Si el smoke falla por una partición sin alguna clase
(datos diminutos), subir `n_extra` del fixture `cocoa_raw_large`, no relajar el código.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q
git add 02-semestre-2/01-vision-por-computador/proyecto 02-semestre-2/01-vision-por-computador/tests
git commit -m "feat(vision): experimentos del cacao (run.py) y demo de diagnóstico"
```

---

### Task 12: Datos reales, resultados e informes

**Files:**
- Modify: `M/data/download.py` (fijar `SHA256_ZIP`), `M/README.md`, `M/proyecto/README.md`,
  `README.md` (raíz, fila de Visión por computador)
- Create: `M/proyecto/resultados/` (`metrics.json`, `confusion.png`, `recall_m1.png`,
  `errores_gradcam.png`, un `diagnostico_*.png` de ejemplo)

- [ ] **Step 1: Descargar y preparar** (6,2 GB; en segundo plano, ~10–30 min según la red)

Run: `uv run python 02-semestre-2/01-vision-por-computador/data/download.py`
Expected: `SHA-256 del zip: <64 hex>` y un resumen con `images` ≈ 1 910 (1 913 fotos únicas menos
las que el JSON nombra sin archivo), `pods` ≈ 2 118, `missing` con pocos nombres, `size_mismatch` y
`exif_orientations`. Anotar las cifras exactas en el ledger.

- [ ] **Step 2: Verificar la orientación (Review Focus 1)**

Generar en el scratchpad un mosaico de 12 fotos preparadas al azar (semilla 42) con sus cajas de
`pods.csv` dibujadas, más 6 de las fotos con EXIF ≠ 1 si las hay, y mirarlo (herramienta Read sobre
el PNG). Si las cajas no caen sobre las mazorcas en las fotos con EXIF ≠ 1, volver a preparar con
`prepare(raw, prepared, apply_exif=True)`, repetir el mosaico y registrar `Ruling:` en el ledger.

- [ ] **Step 3: Fijar el SHA-256**

En `M/data/download.py`, reemplazar `SHA256_ZIP = None  # se fija tras la primera descarga verificada por MD5`
por `SHA256_ZIP = "<el valor impreso en el Step 1>"`.

- [ ] **Step 4: Comprobar las ráfagas**

Run (en el scratchpad): cargar `images.csv`, `assign_bursts`, imprimir número de ráfagas, tamaño
medio y la tabla `pd.crosstab(images.camera, images.folder)`.
Expected: ≈ 109 ráfagas con hora más las numeradas y sueltas; la tabla reproduce la confusión
clase × celular de la spec §2.1.

- [ ] **Step 5: Ejecutar los experimentos** (en segundo plano; ~1 h en un M4)

Run: `uv run python 02-semestre-2/01-vision-por-computador/proyecto/run.py`
Expected: tres líneas `H1/H2/H3: Δ … p_Holm … → apoya: …` y `proyecto/resultados/metrics.json`.
Revisar que `datos.m4_excluidas` ≈ 38, `fotos_por_parte_rafaga` ≈ 70/10/20 y que ninguna métrica sea
nan. Si un número sorprende (p. ej. recall de m1 ≈ 1 en todos los modelos, o AP50 > 0,99), investigar
antes de escribir (systematic-debugging) y registrar lo encontrado.

- [ ] **Step 6: Demo**

Run: `uv run python 02-semestre-2/01-vision-por-computador/proyecto/diagnosticar.py <una foto m1 de test> --salida 02-semestre-2/01-vision-por-computador/proyecto/resultados/diagnostico_ejemplo.png`
Expected: líneas `mazorca N: …` y la figura. Elegir la foto entre las de test por ráfaga (no vistas).

- [ ] **Step 7: Informes**

`M/proyecto/README.md` (reemplaza la plantilla), con las cifras de `metrics.json` y nada que no esté
ahí:
- Problema (monilia: 30–40 % de pérdidas; por qué importa m1) y datos (CocoaMoniliaDataSet v2,
  atribución CC BY 4.0 con DOI del artículo y de Zenodo; fotos, mazorcas, m4 excluidas).
- El hallazgo clase × celular (tabla) y las ráfagas; las tres particiones.
- Tabla de clasificación por ráfaga (4 modelos: recall m1 @ FPR 10 %, FPR h0 real en test,
  F1-macro); partición publicada; control de celular; sonda de celular.
- Tabla de hipótesis (Δ, IC, p, p Holm, ¿se apoya?).
- Detector (AP50, AP50:95) y sistema completo (recall m1, FPR h0, detectadas por clase).
- Figuras (`confusion.png`, `recall_m1.png`, `errores_gradcam.png`, demo).
- Limitaciones (spec §10) y el contexto del 0,91 publicado (otra métrica y partición).
- Uso (`download.py`, `run.py`, `diagnosticar.py`) y la nota de que la aumentación es el TODO del
  usuario (los resultados usan la versión mínima).

`M/README.md` (reemplaza la plantilla, mismo formato que `01-semestre-1/05-aprendizaje-profundo/README.md`):
objetivos (4–5, en «Al terminar el módulo se debe poder:»), tabla de temas → notebook/ejercicio,
estructura de carpetas, cómo trabajar el módulo, recursos y checklist marcado. Recursos:
- R. Szeliski, *Computer Vision: Algorithms and Applications*, 2.ª ed. (2022), libre en szeliski.org/Book.
- S. Ren et al., «Faster R-CNN» (NeurIPS 2015); A. Howard et al., «Searching for MobileNetV3» (ICCV 2019).
- K. He et al., «Deep Residual Learning» (CVPR 2016).
- O. Ronneberger et al., «U-Net» (MICCAI 2015).
- R. Selvaraju et al., «Grad-CAM» (ICCV 2017).
- R. Geirhos et al., «Shortcut learning in deep neural networks» (Nature Machine Intelligence, 2020).
- T. Ojala et al., «Multiresolution gray-scale and rotation invariant texture classification with
  local binary patterns» (TPAMI 2002); N. Otsu (1979); N. Bodla et al., «Soft-NMS» (ICCV 2017).
- CocoaMoniliaDataSet, *Data in Brief* (2025), DOI 10.1016/j.dib.2025.112447.

`README.md` raíz: fila de Visión por computador →
`| S2 | [Visión por computador](02-semestre-2/01-vision-por-computador) | 2 | Propia | Detección temprana de monilia en cacao (CocoaMoniliaDataSet) | 🚧 |`
(se marca ✅ cuando el CI pase en main).

- [ ] **Step 8: Verificación completa y commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest -q && uv run python scripts/run_notebooks.py
git add 02-semestre-2/01-vision-por-computador README.md
git commit -m "docs(vision): resultados del cacao, informe del proyecto y README del módulo"
```

---

## Learn by Doing (al cerrar el módulo)

El TODO de `train_augmentation()` en `applied_cv/cocoa/classifiers.py` es la contribución del usuario:
5–10 líneas que deciden qué transformaciones conservan el síntoma. El test
`test_train_augmentation_preserves_hue` vigila la regla de dominio (no cambiar el tono). Tras su
versión, volver a ejecutar `run.py` y comparar el recall de m1 con el de la versión mínima.
