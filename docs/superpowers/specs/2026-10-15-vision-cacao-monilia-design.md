# Visión por computador (S2) · Detección temprana de monilia en cacao — diseño

> Estado: aprobado por secciones en conversación el 2026-10-15; pendiente de revisión escrita.

## 1. Propósito

Módulo S2 «Visión por computador» con un mini-proyecto sobre un **problema agrícola real de
Colombia**: la moniliasis del cacao (*Moniliophthora roreri*), que causa pérdidas del 30–40 %.

Pregunta central: **¿se distingue una mazorca en etapa temprana (m1, «jorobas», aún tratable) de una
sana, y cuánto de ese acierto es diagnóstico real y no un atajo (fotos repetidas, celular)?**

Lo dijo el usuario: dominio agro, problema real/local, dataset de cacao, foco en detección temprana,
enfoque de dos etapas (detector → clasificador). Supuestos míos: el resto de este documento.

## 2. Datos

**CocoaMoniliaDataSet v2** — Zenodo 17716661 (DOI 10.5281/zenodo.17716661), licencia **CC BY 4.0**,
publicado 2025-11-25. Artículo: *Data in Brief*, DOI 10.1016/j.dib.2025.112447.

- 1 953 fotos de celular (8–12 MP) en dos fincas: Granja Yariguíes (Barrancabermeja, Santander) y
  Bosqueadentro (San Luis, Antioquia); 2 123 mazorcas anotadas.
- Clases: h0 sana (754), m1 jorobas (442), m2 manchas café/aceitosas (416), m3 esporulación (511).
- Anotaciones: polígonos (COCO 1.0 JSON, un archivo por clase), cajas YOLO, máscaras PNG.
- Archivos: `CocoaMoniliaDataSet.zip` (6 187 303 002 bytes, md5 d3531c4a62f3aaff9c146c7a8ae39a6b) y
  listas `train.txt` (1 559), `val.txt` (194), `test.txt` (194) = 1 947 rutas.
- Baseline publicado en el repo del autor (YOLO11l, 640 px): mAP50 0,91; partición no documentada.

### 2.1 Hallazgo previo: confusión clase × celular

El formato del nombre de archivo identifica (aproximadamente) el celular:

| Formato | h0 | m1 | m2 | m3 |
|---|---|---|---|---|
| `IMG_AAAAMMDD_…` | 422 | 272 | 396 | 465 |
| epoch en ms (13 dígitos) | 224 | 8 | 0 | 0 |
| `AAAAMMDD_HHMMSS` | 0 | 153 | 5 | 0 |

Las fotos cubren 9 días con mezclas de clase muy distintas. Un modelo puede distinguir h0 de m1 por
el celular. Esto motiva las particiones y la H3.

### 2.2 Descarga y preparación (`data/download.py` + caché)

- Descarga con verificación MD5 (Zenodo) y SHA-256 propio fijado en el script; reintento reanudable.
- Extracción única y dos derivados en `data/` (ignorado por git):
  - imágenes con lado máximo 1 024 px y cajas COCO reescaladas (detector);
  - recortes por mazorca a 224 px con 10 % de margen, recortado al borde (clasificador).
- Metadatos por imagen: clase, finca si es deducible, formato de celular, marca de tiempo, ráfaga.
- El zip puede borrarse después; el script informa el espacio liberado.

## 3. Particiones

**Ráfaga** = fotos consecutivas del mismo formato de celular separadas por menos de Δ minutos.
Δ = 2 min por defecto, revisado mirando la distribución de intervalos entre fotos (nunca los
resultados) y fijado antes de entrenar. Las fotos sin marca de tiempo legible son cada una su propia
ráfaga.

1. **Publicada**: las listas del autor (referencia; probablemente aleatoria).
2. **Por ráfaga** (principal): 70/10/20 agrupada por ráfaga, estratificada por clase en lo posible.
3. **Control de celular**: modelos entrenados en la partición por ráfaga, evaluados en h0 vs. m1
   solo sobre fotos `IMG_…` (único formato con ambas clases en cantidad).

Diferencias entre fuentes (1 947 rutas frente a 1 953 imágenes, imágenes sin anotación) se reportan,
no se corrigen inventando datos.

## 4. Modelos

**Detector de mazorcas** (una sola clase): Faster R-CNN MobileNetV3-Large-FPN de torchvision con pesos
COCO (BSD), fine-tuning en MPS con caída a CPU (con aviso) si un operador no está soportado.

**Clasificador de etapa** sobre recortes:

| Modelo | Rol |
|---|---|
| Prevalencia | piso |
| Color + textura: histograma HSV + LBP (desde cero) + logística | ¿basta «color café y rugosidad»? |
| ResNet-18 ImageNet, sonda lineal (congelada) | valor de ImageNet sin adaptar |
| ResNet-18 ImageNet, fine-tuning | modelo fuerte |

Aumentación: volteos, rotaciones, brillo y contraste; **nunca tono** (el color es el síntoma). La
política concreta es la contribución del usuario (TODO, §7).

Se descarta `ultralytics`/YOLO por su licencia AGPL (el repo es MIT y público).

## 5. Métricas e hipótesis

**Métrica central**: recall de m1 con FPR en h0 = 10 %, con puntaje P(enferma) = P(m1)+P(m2)+P(m3) y
umbral fijado en validación.

**Secundarias**: F1-macro y matriz de confusión de 4 clases; detector AP50 y AP50:95 (una clase);
sistema completo por imagen, donde una m1 cuenta como hallada si su caja predicha tiene IoU ≥ 0,5 y
la etapa predicha es enferma.

**Hipótesis** (antes de ver test; IC 95 % por bootstrap por ráfaga, 2 000 réplicas; Holm sobre las
tres; «apoya» = rechaza H0 ∧ diferencia en la dirección prevista):

- **H1**: fine-tuning > color + textura en recall de m1 @ FPR 10 % (partición por ráfaga; pareado).
- **H2**: F1-macro(partición publicada) − F1-macro(partición por ráfaga) > 0 (ResNet-18 con fine-tuning, misma receta;
  bootstrap no pareado, conjuntos de test distintos).
- **H3**: dentro de h0 únicamente, una sonda lineal sobre ResNet-18 predice el formato de celular
  (`IMG_…` vs. epoch) con exactitud balanceada > 0,5 (validación cruzada agrupada por ráfaga).

No se compara directamente con el 0,91 publicado (otra métrica, otra tarea, partición desconocida);
se cita como contexto.

## 6. Módulo

Plantilla de S1 (`01-semestre-1/01-matematicas-aplicadas`). Teoría en notebooks smoke, sin descargas
ni pesos preentrenados, en segundos:

| Notebook | Contenido (desde cero) |
|---|---|
| 01 Imágenes, color y filtros | convolución 2D, Sobel, RGB→HSV, histogramas, LBP |
| 02 Transfer learning | CNN chica preentrenada en tarea sintética A → tarea B vs. desde cero; sonda vs. fine-tuning |
| 03 Detección | IoU, NMS, anclas, curva PR y AP |
| 04 Segmentación | U-Net pequeña en figuras sintéticas; Dice e IoU |
| 05 Interpretabilidad y atajos | Grad-CAM; red que aprende una marca de agua en vez del objeto |

Paquete `src/applied_cv/` (registrado en `pyproject.toml`):

- `image.py`: conv2d, Sobel, rgb_to_hsv, histogramas, LBP.
- `boxes.py`: IoU, NMS, AP.
- `masks.py`: Dice, IoU.
- `unet.py`: U-Net pequeña.
- `gradcam.py`: Grad-CAM.
- `cocoa/`: lectura COCO, ráfagas, particiones, recortes, modelos y evaluación del proyecto.

Ejercicios (soluciones aparte, `check()`): interpolación bilineal, IoU en casos borde, soft-NMS,
tamaño de salida y campo receptivo, umbral de Otsu, RGB→HSV a mano.

Proyecto: `proyecto/run.py` (experimentos → `proyecto/resultados/`: `metrics.json`, figuras, mosaico
de errores con Grad-CAM) y un script de demo: foto → imagen anotada con cajas, etapa y Grad-CAM.

Dependencias: `scikit-image` se agrega al extra `dl` solo como referencia de tests (con
`pytest.importorskip`); AP sin `pycocotools`, verificado con casos calculados a mano.

## 7. Contribución del usuario

TODO en el código del proyecto: la política de aumentación del clasificador (qué transformaciones
conservan el síntoma). Se deja una versión mínima funcional que el usuario reemplaza.

## 8. Casos borde

- Imágenes sin anotación: excluidas y contadas. Imágenes con varias mazorcas: evaluación por mazorca.
- Nombres sin fecha (`m0 (95).jpg`, etc.): ráfaga propia cada uno.
- Recortes con margen fuera del borde: recortados a la imagen.
- MPS sin soporte de un operador: CPU con aviso; semillas fijas.
- Demo sin mazorcas detectadas: lo dice; no inventa diagnóstico.
- Demo con archivo inexistente o no imagen: mensaje claro, código de salida ≠ 0.

## 9. Pruebas

- Funciones desde cero contra referencias: `torchvision.ops.box_iou`/`nms`, `skimage` (LBP, HSV),
  AP a mano; Dice/IoU con máscaras conocidas; Grad-CAM con una red cuya respuesta se conoce.
- Fixture sintético (pocas imágenes chicas + COCO JSON + listas de partición) para carga, ráfagas,
  particiones, recortes y `run.py` en modo smoke.
- El CI no descarga datos ni pesos.

## 10. Limitaciones (se declararán)

Dos fincas y 9 días: sin validación en fincas nuevas. El celular está confundido con la clase y solo
se controla en parte. m1 es una etapa visual (jorobas), sin confirmación de laboratorio.

## 11. Criterios de éxito

1. H1–H3 respondidas con IC y Holm, aunque salgan en contra.
2. La brecha entre partición publicada y por ráfaga, cuantificada.
3. Un sistema de punta a punta que procesa una foto nueva.
4. Cada cifra del README trazable a `metrics.json`.
