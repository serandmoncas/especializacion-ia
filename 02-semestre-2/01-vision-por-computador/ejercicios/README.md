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
