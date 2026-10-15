# Ejercicios — Sistemas de recomendación

Cada ejercicio tiene un archivo `ejNN_*.py` con firmas, docstrings y una función `check()`. Usan el
generador sintético de `applied_recsys.synthetic`, que conoce los gustos reales. Implementar y
ejecutar `uv run python ejercicios/ejNN_*.py` desde la carpeta del sub-módulo.
Las soluciones están en [`soluciones/`](soluciones/).

| # | Tema | Archivo |
|---|---|---|
| 1 | BPR: entrenar directamente para rankear | `ej01_bpr.py` |
| 2 | Arranque en frío: ítems nuevos a partir de sus atributos | `ej02_cold_start.py` |
| 3 | Diversificación con MMR | `ej03_mmr.py` |
| 4 | Corregir la evaluación offline con IPS | `ej04_ips.py` |
| 5 | (Papel) ALS, confianza, NDCG y una pregunta abierta | abajo |

## 1. BPR

Bayesian Personalized Ranking (Rendle et al., 2009) optimiza el orden: para tripletas (u, i visto,
j no visto) maximiza $\log\sigma(\hat x_{ui} - \hat x_{uj})$. Implementarlo con SGD por minilotes sobre
factores latentes y comparar el AUC en test con el de popularidad y el de FunkSVD.

## 2. Arranque en frío

Un ítem nuevo no tiene calificaciones, así que el filtrado colaborativo no puede ubicarlo. Si tiene
atributos (género, director…), se puede aprender una función atributos → factores latentes con los
ítems conocidos (ridge) y aplicarla al nuevo. Medir el RMSE sobre ítems fríos frente al baseline
que ignora el ítem (μ + b_u) y frente a los ítems con historia.

## 3. MMR

Maximal Marginal Relevance (Carbonell & Goldstein, 1998) arma la lista de forma voraz:
$\arg\max_i\ \lambda\,\text{rel}(i) - (1-\lambda)\max_{j \in S} \text{sim}(i, j)$. Implementarlo y
graficar la curva relevancia total vs. diversidad intra-lista al variar λ.

## 4. IPS

El notebook 03 muestra que la evaluación offline puede invertir el orden de los modelos: lo
observado depende de la exposición. *Inverse propensity scoring* pondera cada acierto por
$1/P(\text{observado})$. Con el generador sintético (que conoce esas probabilidades) mostrar que IPS
estima sin sesgo la precisión real de cada modelo, mientras el estimador ingenuo no. Ojo: un ítem
fuera del train cae en test con probabilidad $p_{test} = 0.25\,p / (1 - 0.75\,p)$, no $0.25\,p$.
Discutir la varianza (¿qué pasa si $p$ es diminuta?) y qué ocurre si la propensión solo se puede
*estimar* con la popularidad.

## 5. Papel

(a) Derivar la actualización de ALS explícito para $p_u$ con $q$ fijo.
(b) ¿Por qué ALS implícito no equivale a PureSVD aunque ambos tratan los faltantes como 0?
(c) Mostrar que NDCG@k ∈ [0, 1] y que vale 1 sii los primeros min(k, |rel|) puestos son relevantes.
(d) **Pregunta abierta del proyecto.** En MovieLens, la popularidad alcanza a los modelos para los
usuarios más activos (Q4), aunque sus relevantes son *menos* populares (12 % en el top-100 frente
a 28 % en Q1). Proponer hipótesis y un experimento para distinguirlas.
