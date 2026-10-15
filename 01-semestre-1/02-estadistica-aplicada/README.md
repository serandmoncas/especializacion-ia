# Estadística aplicada

> Semestre 1 · 2 créditos · Compartida

## Objetivos

Al terminar el módulo se debe poder:

1. Decidir **por simulación** si una aproximación (normal, intervalo t) es confiable para unos datos dados.
2. Diseñar un experimento: métrica, tamaño de muestra, potencia y plan de análisis preregistrado.
3. Analizarlo con métodos paramétricos, de remuestreo y bayesianos, y explicar cuándo difieren.
4. Reconocer y evitar las trampas habituales: peeking, comparaciones múltiples, SRM,
   “no significativo = no hay efecto”.

## Temas

| Tema | Dónde |
|---|---|
| LGN, TLC, cobertura de intervalos según la distribución | `notebooks/01-probabilidad-y-simulacion.ipynb` |
| p-valores, potencia, tamaño de muestra | `notebooks/02-pruebas-de-hipotesis.ipynb` |
| Peeking, comparaciones múltiples (Holm, BH) | `notebooks/02-pruebas-de-hipotesis.ipynb` |
| Bootstrap (y cuándo falla), permutación | `notebooks/03-bootstrap-permutacion-bayes.ipynb` |
| A/B bayesiano: Beta–Binomial, P(B > A), pérdida esperada, priors | `notebooks/03-bootstrap-permutacion-bayes.ipynb` |
| Wilson, BCa, Pocock, Thompson sampling, CUPED | `ejercicios/` |
| SRM, balance, CUPED, análisis completo de un experimento real | `proyecto/` |

## Estructura

```
02-estadistica-aplicada/
├── notebooks/            # 01 simulación · 02 pruebas y trampas · 03 bootstrap/permutación/Bayes
├── src/applied_stats/    # inference, resampling, bayes, multiple, experiments
├── tests/                # cada función contra scipy o por simulación + soluciones
├── ejercicios/           # 4 en código + 1 en papel; soluciones/ aparte
├── data/download.py      # Hillstrom (MineThatData), con checksum SHA-256
└── proyecto/             # análisis A/B/C de Hillstrom: run.py + resultados/
```

## Cómo trabajar el módulo

```bash
uv sync --extra base
uv run jupyter lab 01-semestre-1/02-estadistica-aplicada/notebooks
uv run pytest 01-semestre-1/02-estadistica-aplicada
```

## Recursos

- L. Wasserman, *All of Statistics* (2004): caps. 6–11 (inferencia, bootstrap, pruebas).
- B. Efron, R. Tibshirani, *An Introduction to the Bootstrap* (1993): caps. 13–14 (BCa).
- R. Kohavi, D. Tang, Y. Xu, *Trustworthy Online Controlled Experiments* (2020): SRM, CUPED,
  peeking; la referencia práctica de A/B testing.
- A. Deng et al., “Improving the Sensitivity of Online Controlled Experiments by Utilizing
  Pre-Experiment Data” (WSDM 2013): el artículo de CUPED.
- A. Gelman et al., *Bayesian Data Analysis*, 3.ª ed. (2013): cap. 2 (modelos conjugados).
- K. Hillstrom, [MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) (2008).

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (3, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (5; soluciones verificadas por tests)
- [x] Mini-proyecto con README, código y resultados (decisión final: pendiente, ver proyecto)
- [x] Tests de las funciones en `src/`
- [x] CI en verde
