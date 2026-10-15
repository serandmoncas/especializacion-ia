# Ejercicios — Matemáticas aplicadas

Cada ejercicio tiene un archivo `ejNN_*.py` con la firma, un docstring y una función `check()`.
Implementar, luego ejecutar `uv run python ejercicios/ejNN_*.py` desde la carpeta del módulo.
Las soluciones están en [`soluciones/`](soluciones/): intentarlo antes de mirarlas.

| # | Tema | Archivo |
|---|---|---|
| 1 | Gradiente y forma cerrada de Ridge | `ej01_ridge.py` |
| 2 | Gradiente de la entropía cruzada (regresión logística) | `ej02_logistic.py` |
| 3 | Búsqueda lineal con backtracking (Armijo) | `ej03_backtracking.py` |
| 4 | Método de Newton vs. GD en Rosenbrock | `ej04_newton.py` |
| 5 | (Papel) Condición de estabilidad de GD | abajo |

## 1. Ridge

$L(w) = \frac{1}{n}\|Xw - y\|^2 + \lambda\|w\|^2$. Derivar $\nabla L$ y la solución cerrada
$w^* = (X^\top X + n\lambda I)^{-1}X^\top y$. Implementar ambos y verificar con `gradient_check`.
¿Por qué Ridge siempre tiene solución única aunque $X^\top X$ sea singular?

## 2. Regresión logística

Con $p = \sigma(Xw)$, $L(w) = -\frac1n \sum_i y_i\log p_i + (1-y_i)\log(1-p_i)$.
Demostrar que $\nabla L = \frac1n X^\top(p - y)$ (la misma forma que el MSE, sin el 2).
Implementar una versión numéricamente estable de la pérdida (pista: `np.logaddexp`).

## 3. Backtracking (Armijo)

Partiendo de $\eta_0$, reducir $\eta \leftarrow \rho\eta$ hasta que
$f(x - \eta\nabla f) \le f(x) - c\,\eta\,\|\nabla f\|^2$. Integrarlo en un GD y comparar el
número de iteraciones con GD de paso fijo en la cuadrática del notebook 03.

## 4. Newton

$x \leftarrow x - H^{-1}\nabla f$. Implementar para Rosenbrock
$f(x,y) = (1-x)^2 + 100(y-x^2)^2$ desde $(-1.2, 1)$ y comparar iteraciones con GD.
¿Por qué Newton es invariante a reescalamientos lineales de las variables y GD no?

## 5. Estabilidad de GD (papel)

Para $f(x) = \frac12 x^\top H x$ con $H \succ 0$, demostrar que GD converge desde todo $x_0$
si y solo si $0 < \eta < 2/\lambda_{max}(H)$, y que el factor de contracción óptimo es
$\frac{\kappa - 1}{\kappa + 1}$ con $\eta^* = \frac{2}{\lambda_{min} + \lambda_{max}}$.
