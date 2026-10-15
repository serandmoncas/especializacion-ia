# Ejercicios — Series de tiempo

Cada ejercicio tiene un archivo `ejNN_*.py` con firmas, docstrings y una función `check()`.
Implementar y ejecutar `uv run python ejercicios/ejNN_*.py` desde la carpeta del sub-módulo.
Las soluciones están en [`soluciones/`](soluciones/).

| # | Tema | Archivo |
|---|---|---|
| 1 | Intervalos de predicción de un AR(1): fórmula vs. cobertura empírica | `ej01_intervals.py` |
| 2 | Prueba de Diebold–Mariano para comparar pronósticos | `ej02_diebold_mariano.py` |
| 3 | ARMA(1,1) por mínimos cuadrados condicionales | `ej03_arma.py` |
| 4 | Detección de anomalías con residuos robustos | `ej04_anomalies.py` |
| 5 | (Papel) Caminata aleatoria, MASE, SES ≡ ARIMA(0,1,1) | abajo |

## 1. Intervalos de un AR(1)

Para $y_t = c + \phi y_{t-1} + \varepsilon_t$, el error de pronóstico a $h$ pasos tiene varianza
$\sigma^2 \sum_{j=0}^{h-1} \phi^{2j}$. Construir el intervalo del 95 % y medir su cobertura por
simulación. ¿Qué pasa si se ignora la incertidumbre de haber *estimado* $\phi$ con pocos datos?

## 2. Diebold–Mariano

Comparar dos pronósticos con un t-test sobre las diferencias de pérdida $d_t$ ignora que, a
horizonte $h$, los $d_t$ están autocorrelacionados (hasta el rezago $h-1$). DM usa una varianza
de largo plazo (autocovarianzas hasta $h-1$). Implementarla y verificar que, cuando ambos
pronósticos son igual de buenos, rechaza ≈ 5 % de las veces.

## 3. ARMA(1,1)

$y_t = c + \phi y_{t-1} + \varepsilon_t + \theta \varepsilon_{t-1}$. Los errores no se observan:
dado $(c, \phi, \theta)$ se reconstruyen recursivamente ($\varepsilon_0 = 0$) y se minimiza su suma
de cuadrados. Comparar con `statsmodels.tsa.arima.model.ARIMA`.

## 4. Anomalías

Restar un pronóstico estacional robusto (mediana de las últimas k semanas a la misma hora) y
marcar residuos con $|r| > 4 \cdot \text{MAD}$. Aplicarlo a Bike Sharing: ¿aparecen los cierres
por huracán? ¿Por qué se detectan los de día y no los de madrugada? ¿Por qué mediana y MAD en vez
de media y desviación estándar?

## 5. Papel

(a) Si $y_t$ es una caminata aleatoria, mostrar que el pronóstico naive es óptimo y que la varianza
del error a $h$ pasos es $h\sigma^2$.
(b) ¿Por qué MASE < 1 no garantiza que un modelo supere al seasonal naive *en el test*?
(c) Mostrar que el suavizamiento exponencial simple produce los mismos pronósticos que un
ARIMA(0,1,1) con $\theta = \alpha - 1$.
