# Matemáticas aplicadas

> Semestre 1 · 2 créditos · Compartida
> **Módulo de referencia:** los demás módulos siguen esta estructura.

## Objetivos

Al terminar el módulo se debe poder:

1. Interpretar mínimos cuadrados como una proyección ortogonal y PCA como una SVD.
2. Derivar gradientes de pérdidas comunes y **verificarlos numéricamente**.
3. Predecir el comportamiento de GD a partir de la curvatura (L, κ) y elegir η con criterio.
4. Implementar GD, momentum y SGD por minilotes en NumPy y validarlos contra scikit-learn.

## Temas

| Tema | Dónde |
|---|---|
| Vectores, proyecciones, espacio columna | `notebooks/01-algebra-lineal.ipynb` |
| Autovalores, iteración de potencias, SVD, PCA | `notebooks/01-algebra-lineal.ipynb` |
| Diferencias finitas, gradient check | `notebooks/02-calculo-y-gradientes.ipynb` |
| Gradiente del MSE, Hessiana, condicionamiento | `notebooks/02-calculo-y-gradientes.ipynb` |
| Tasa de aprendizaje, momentum, estandarización, SGD | `notebooks/03-optimizacion.ipynb` |
| Ridge, logística, Armijo, Newton | `ejercicios/` |

## Estructura

```
01-matematicas-aplicadas/
├── notebooks/           # 01 álgebra lineal · 02 gradientes · 03 optimización
├── src/applied_math/    # linalg, gradients, optim, linear_regression (importable: `import applied_math`)
├── tests/               # pytest de src/ y de las soluciones
├── ejercicios/          # enunciados + stubs; soluciones/ aparte
├── data/download.py     # California Housing → data/california_housing.csv
└── proyecto/            # regresión lineal desde cero: run.py + resultados/
```

## Cómo trabajar el módulo

```bash
uv sync --extra base
uv run jupyter lab 01-semestre-1/01-matematicas-aplicadas/notebooks
uv run pytest 01-semestre-1/01-matematicas-aplicadas
```

## Recursos

- G. Strang, *Linear Algebra and Learning from Data* (2019) — caps. I y VI.
- M. P. Deisenroth, A. A. Faisal, C. S. Ong, *Mathematics for Machine Learning* (2020) —
  libre en [mml-book.github.io](https://mml-book.github.io/). Caps. 2–7.
- 3Blue1Brown, *Essence of Linear Algebra* (serie de video) — intuición geométrica.
- S. Boyd, L. Vandenberghe, *Convex Optimization* (2004) — cap. 9 (descenso, Newton, backtracking).
- G. Goh, [*Why Momentum Really Works*](https://distill.pub/2017/momentum/), Distill (2017).

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (3, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (5; soluciones verificadas por tests)
- [x] Mini-proyecto con README, código y resultados
- [x] Tests de las funciones en `src/`
- [ ] CI en verde (pendiente del primer push)
