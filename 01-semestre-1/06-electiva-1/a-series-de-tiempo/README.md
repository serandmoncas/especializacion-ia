# Series de tiempo

> Electiva 1 (S1) · sub-módulo A

## Objetivos

Al terminar el sub-módulo se debe poder:

1. Descomponer una serie, leer su ACF/PACF y decidir cómo volverla estacionaria.
2. Implementar baselines, suavizamiento exponencial y modelos AR; usar ARIMA/SARIMA de statsmodels.
3. Convertir el pronóstico en un problema tabular con rezagos, eligiendo entre estrategia recursiva y directa.
4. Evaluar pronósticos con backtesting de origen móvil y compararlos de forma estadísticamente honesta.

## Temas

| Tema | Dónde |
|---|---|
| Descomposición, ACF/PACF, estacionariedad, ADF, correlaciones espurias | `notebooks/01-componentes-y-autocorrelacion.ipynb` |
| Baselines, Holt–Winters, AR, SARIMA, intervalos | `notebooks/02-suavizamiento-y-arima.ipynb` |
| K-fold vs. validación temporal, rezagos, recursivo vs. directo | `notebooks/03-validacion-temporal-y-ml.ipynb` |
| Intervalos de AR, Diebold–Mariano, ARMA, anomalías | `ejercicios/` |
| Pronóstico del día siguiente con backtesting | `proyecto/` |

## Estructura

```
a-series-de-tiempo/
├── notebooks/           # 01 componentes · 02 suavizamiento y ARIMA · 03 validación y ML
├── src/applied_ts/
│   ├── decomposition.py # media móvil centrada, descomposición clásica
│   ├── autocorrelation.py  # ACF, PACF (Durbin–Levinson), diferenciación
│   ├── models.py        # baselines, Holt–Winters, AR por mínimos cuadrados
│   ├── features.py      # rezagos y calendario para ML
│   ├── evaluation.py    # MAE, RMSE, sMAPE, MASE, origen móvil, backtest
│   └── bikes.py         # UCI Bike Sharing como serie horaria completa
├── tests/               # contra statsmodels (descomposición, ACF/PACF, AutoReg, Holt–Winters)
├── ejercicios/          # 4 en código + 1 en papel; soluciones/ aparte
├── data/download.py     # UCI Bike Sharing, con checksum
└── proyecto/            # 6 modelos, 184 orígenes, comparación pareada
```

## Cómo trabajar el sub-módulo

```bash
uv sync --extra base
uv run jupyter lab 01-semestre-1/06-electiva-1/a-series-de-tiempo/notebooks
uv run pytest 01-semestre-1/06-electiva-1/a-series-de-tiempo
```

Los notebooks usan el CO₂ de Mauna Loa incluido en statsmodels: no requieren descargas.

## Recursos

- R. J. Hyndman, G. Athanasopoulos, *Forecasting: Principles and Practice*, 3.ª ed. (2021):
  libre en otexts.com/fpp3. La referencia práctica; caps. 3, 5, 8, 9.
- G. Box, G. Jenkins, G. Reinsel, G. Ljung, *Time Series Analysis: Forecasting and Control*, 5.ª ed. (2015).
- R. J. Hyndman, A. B. Koehler, “Another look at measures of forecast accuracy” (IJF, 2006): MASE.
- F. X. Diebold, R. S. Mariano, “Comparing Predictive Accuracy” (JBES, 1995).
- S. Makridakis et al., competencias M4/M5 (IJF, 2020–2022): qué funciona en la práctica.

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (3, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (5; soluciones verificadas por tests)
- [x] Mini-proyecto con README, código y resultados (intervalos de predicción: tu turno)
- [x] Tests de las funciones en `src/`
- [ ] CI en verde
