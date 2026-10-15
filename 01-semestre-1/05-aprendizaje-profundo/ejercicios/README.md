# Ejercicios — Aprendizaje profundo

Cada ejercicio tiene un archivo `ejNN_*.py` con firmas, docstrings y una función `check()` que
compara contra PyTorch. Implementar y ejecutar `uv run python ejercicios/ejNN_*.py` desde la
carpeta del módulo. Las soluciones están en [`soluciones/`](soluciones/).

| # | Tema | Archivo |
|---|---|---|
| 1 | Max pooling con backward | `ej01_maxpool.py` |
| 2 | Batch normalization (entrenamiento y evaluación) | `ej02_batchnorm.py` |
| 3 | Atención con producto escalar escalado | `ej03_attention.py` |
| 4 | Campo receptivo de una CNN | `ej04_receptive_field.py` |
| 5 | (Papel) Gradientes, inicialización, LSTM, parámetros | abajo |

## 1. Max pooling

Implementar `max_pool2d(x, k)` (ventanas k×k sin solapamiento) como operación del autograd de
`applied_dl`. En el backward, el gradiente de cada ventana va **solo** a la posición del máximo.

## 2. Batch normalization

En entrenamiento: normalizar cada canal con la media y varianza *del lote*, luego $\gamma \hat x + \beta$,
y actualizar promedios móviles. En evaluación: usar los promedios móviles. Construirlo con
operaciones del `Tensor` (así el backward sale solo) y comparar salida y gradientes con
`torch.nn.functional.batch_norm`. Ojo: PyTorch normaliza con la varianza sesgada pero guarda la
insesgada en el promedio móvil.

## 3. Atención

$\text{Attention}(Q, K, V) = \text{softmax}(QK^\top/\sqrt{d})\,V$, con máscara causal opcional
(cada posición solo ve las anteriores). Comparar con `torch.nn.functional.scaled_dot_product_attention`.
Luego medir por qué se divide por $\sqrt d$: varianza de $q \cdot k$ y entropía del softmax con y
sin escala, para $d$ = 8, 64, 512.

## 4. Campo receptivo

¿Cuántos pixeles de la entrada pueden influir en una unidad de la última capa convolucional de la
CNN del proyecto? Calcularlo con la recurrencia $r_l = r_{l-1} + (k_l - 1)\,j_{l-1}$,
$j_l = j_{l-1} s_l$, y verificarlo empíricamente: el gradiente de una unidad central respecto a
la entrada solo es distinto de cero dentro del campo receptivo.

## 5. Papel

(a) Demostrar que $\partial\, \text{CE}(\text{softmax}(z), y)/\partial z = \text{softmax}(z) - \text{onehot}(y)$.
(b) Con $h = \text{ReLU}(Wx)$, entradas de varianza 1 y $W_{ij} \sim \mathcal N(0, \sigma^2)$, mostrar
que $\sigma^2 = 2/\text{fan\_in}$ preserva $E[h^2]$ capa a capa.
(c) En una LSTM, $c_t = f_t \odot c_{t-1} + i_t \odot g_t$. ¿Qué es $\partial c_t / \partial c_{t-k}$
(ignorando la dependencia de las compuertas)? Relacionarlo con el notebook 04.
(d) Contar a mano los parámetros de la CNN del proyecto (468 202).
