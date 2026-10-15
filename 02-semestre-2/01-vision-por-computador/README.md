# Visión por computador

> Semestre 2 · 2 créditos · Propia · requiere `uv sync --extra base --extra dl`

## Objetivos

Al terminar el módulo se debe poder:

1. Implementar las operaciones clásicas (convolución, Sobel, HSV, LBP) y explicar qué aprende la
   primera capa de una CNN a partir de ellas.
2. Reutilizar una red preentrenada como extractor (sonda lineal) o ajustándola (fine-tuning), y
   saber cuándo conviene cada una.
3. Evaluar detectores (IoU, NMS, AP50 y AP50:95) y segmentadores (Dice, IoU) con implementaciones
   propias verificadas contra torchvision y scikit-image.
4. Usar Grad-CAM para ver dónde mira un modelo y detectar aprendizaje por atajos.
5. Diseñar una evaluación honesta para fotos de campo: particiones agrupadas por ráfaga, control
   de variables confundidas (el celular) e hipótesis con bootstrap por conglomerados y Holm.

## Temas

| Tema | Dónde |
|---|---|
| Convolución, Sobel, RGB→HSV, histogramas, LBP | `notebooks/01-imagenes-color-y-filtros.ipynb` |
| Transfer learning: desde cero vs. sonda lineal vs. fine-tuning | `notebooks/02-transfer-learning.ipynb` |
| IoU, NMS, AP50 vs. AP50:95, anclas | `notebooks/03-deteccion.ipynb` |
| U-Net, Dice e IoU frente a un umbral fijo | `notebooks/04-segmentacion.ipynb` |
| Grad-CAM y una red que aprende una marca de agua | `notebooks/05-interpretabilidad-y-atajos.ipynb` |
| Bilineal, IoU en casos borde, soft-NMS, campo receptivo, Otsu, HSV→RGB | `ejercicios/` |
| Detección temprana de monilia en cacao (detector + clasificador de etapa) | `proyecto/` |

## Estructura

```
01-vision-por-computador/
├── notebooks/            # 01 filtros · 02 transfer · 03 detección · 04 segmentación · 05 Grad-CAM
├── src/applied_cv/
│   ├── image.py          # conv2d, Sobel, HSV, gris, LBP, histogramas (NumPy)
│   ├── boxes.py          # IoU, NMS, AP y AP50:95
│   ├── masks.py          # Dice e IoU de máscaras
│   ├── unet.py           # U-Net pequeña
│   ├── gradcam.py        # Grad-CAM con hooks
│   └── cocoa/            # proyecto: datos, ráfagas, rasgos, clasificadores, detector, evaluación
├── tests/                # contra scikit-image, scipy y torchvision; smoke de run.py y de la demo
├── ejercicios/           # soluciones/ aparte
├── data/download.py      # CocoaMoniliaDataSet v2 (Zenodo, 6,2 GB) con MD5 y SHA-256
└── proyecto/             # run.py (experimentos) y diagnosticar.py (demo)
```

## Cómo trabajar el módulo

```bash
uv sync --extra base --extra dl
uv run jupyter lab 02-semestre-2/01-vision-por-computador/notebooks
uv run pytest 02-semestre-2/01-vision-por-computador
```

Los notebooks usan datos sintéticos y corren en segundos en CPU. El proyecto necesita la descarga
(ver [`proyecto/`](proyecto/)).

## Recursos

- R. Szeliski, *Computer Vision: Algorithms and Applications*, 2.ª ed. (2022). Libre en
  szeliski.org/Book.
- S. Ren et al., «Faster R-CNN» (NeurIPS 2015); A. Howard et al., «Searching for MobileNetV3»
  (ICCV 2019).
- K. He et al., «Deep Residual Learning for Image Recognition» (CVPR 2016).
- O. Ronneberger et al., «U-Net» (MICCAI 2015).
- R. Selvaraju et al., «Grad-CAM» (ICCV 2017).
- R. Geirhos et al., «Shortcut learning in deep neural networks» (Nature Machine Intelligence,
  2020).
- T. Ojala et al., «Multiresolution gray-scale and rotation invariant texture classification with
  local binary patterns» (TPAMI 2002); N. Otsu (1979); N. Bodla et al., «Soft-NMS» (ICCV 2017).
- CocoaMoniliaDataSet, *Data in Brief* (2025), DOI 10.1016/j.dib.2025.112447.

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (5, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (6; verificadas contra torch, torchvision y skimage)
- [x] Mini-proyecto con README, código y resultados (aumentación: tu turno)
- [x] Tests de las funciones en `src/`
- [ ] CI en verde
