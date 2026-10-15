# Ejercicios — Aprendizaje automático

Cada ejercicio tiene un archivo `ejNN_*.py` con firmas, docstrings y una función `check()`.
Implementar y ejecutar `uv run python ejercicios/ejNN_*.py` desde la carpeta del módulo.
Las soluciones están en [`soluciones/`](soluciones/): intentarlo antes de mirarlas.

| # | Tema | Archivo |
|---|---|---|
| 1 | k vecinos más cercanos y la maldición de la dimensión | `ej01_knn.py` |
| 2 | Naive Bayes gaussiano en espacio logarítmico | `ej02_naive_bayes.py` |
| 3 | Regresión isotónica (PAV) para calibrar probabilidades | `ej03_isotonic.py` |
| 4 | Importancia por permutación | `ej04_permutation.py` |
| 5 | (Papel) Sesgo–varianza, AUC, convexidad, OOB | abajo |

## 1. k-NN

Implementar un clasificador k-NN (voto mayoritario, distancia euclidiana, vectorizado).
Luego medir la **concentración de distancias**: para puntos uniformes en $[0,1]^d$, la razón
$(\max - \min)/\min$ de las distancias a un punto de consulta tiende a 0 al crecer $d$.
¿Qué significa “vecino más cercano” cuando todos están casi a la misma distancia?

## 2. Naive Bayes gaussiano

$P(y \mid x) \propto P(y)\prod_j \mathcal N(x_j; \mu_{yj}, \sigma^2_{yj})$. Con cientos de features
el producto de densidades da *underflow* (¿desde qué dimensión? medirlo): trabajar con
log-probabilidades y normalizar con log-sum-exp.
Reproducir exactamente `sklearn.naive_bayes.GaussianNB` (incluido su `var_smoothing`).

## 3. Regresión isotónica

Dado $(s_i, y_i)$ ordenado por $s$, encontrar la función no decreciente $g$ que minimiza
$\sum (y_i - g(s_i))^2$. El algoritmo *Pool Adjacent Violators* recorre los puntos y, cuando un
bloque tiene media mayor que el siguiente, los fusiona. Es lo que usa `CalibratedClassifierCV`
con `method="isotonic"`. Comparar con `sklearn.isotonic.IsotonicRegression`.

## 4. Importancia por permutación

Importancia de la feature $j$ = caída de la métrica al permutar la columna $j$ en datos de
evaluación (rompe su relación con $y$ sin cambiar su distribución). Implementarla y comparar
con `sklearn.inspection.permutation_importance`. ¿Qué pasa con dos features muy correlacionadas?

## 5. Papel

(a) Derivar $\mathbb{E}[(y - \hat f(x))^2] = \sigma^2 + \text{sesgo}^2 + \text{varianza}$.
(b) Demostrar que el AUC (área bajo la ROC) es igual a $P(s^+ > s^-) + \tfrac12 P(s^+ = s^-)$.
(c) Demostrar que la pérdida logística es convexa en $w$.
(d) Mostrar que la fracción de muestras fuera de la bolsa en un bootstrap tiende a $1/e \approx 0.368$.
