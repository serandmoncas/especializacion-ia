# Sistemas de recomendación

> Electiva 1 (S1) · sub-módulo B

## Objetivos

Al terminar el sub-módulo se debe poder:

1. Construir recomendadores por sesgos, vecinos y factorización de matrices (explícita e implícita).
2. Distinguir predecir calificaciones de rankear, y elegir métricas acordes (RMSE vs. NDCG, cobertura,
   sesgo de popularidad).
3. Diseñar una evaluación offline honesta (partición temporal, validación para hiperparámetros) y
   conocer sus límites: el sesgo de exposición y por qué hacen falta experimentos A/B.

## Temas

| Tema | Dónde |
|---|---|
| Cola larga, faltantes no aleatorios (MNAR), sesgos, item-kNN | `notebooks/01-filtrado-colaborativo.ipynb` |
| FunkSVD (SGD), ALS, dimensión latente, recuperación de factores | `notebooks/02-factorizacion-de-matrices.ipynb` |
| Ranking top-N, la trampa de la evaluación offline, ALS implícito | `notebooks/03-ranking-e-implicito.ipynb` |
| BPR, arranque en frío, MMR, IPS | `ejercicios/` |
| Recomendador de películas en MovieLens 1M | `proyecto/` |

## Estructura

```
b-sistemas-de-recomendacion/
├── notebooks/             # 01 filtrado colaborativo · 02 factorización · 03 ranking e implícito
├── src/applied_recsys/
│   ├── data.py            # MovieLens, particiones temporal/aleatoria, matrices dispersas
│   ├── baselines.py       # sesgos regularizados, popularidad
│   ├── neighbors.py       # similitud con encogimiento, item-kNN
│   ├── factorization.py   # FunkSVD, ALS explícito, ALS implícito, PureSVD
│   ├── metrics.py         # RMSE, top-N, precision/recall/NDCG/hit, cobertura, sesgo de popularidad
│   └── synthetic.py       # datos con gusto de bajo rango, cola larga y MNAR (conoce la verdad)
├── tests/                 # contra soluciones exactas (ridge, SVD densa) y sklearn
├── ejercicios/            # 4 en código + 1 en papel; soluciones/ aparte
├── data/download.py       # MovieLens 1M (no redistribuible), con checksum
└── proyecto/              # 5 modelos de rating, 6 de ranking, por actividad del usuario
```

## Cómo trabajar el sub-módulo

```bash
uv sync --extra base
uv run jupyter lab 01-semestre-1/06-electiva-1/b-sistemas-de-recomendacion/notebooks
uv run pytest 01-semestre-1/06-electiva-1/b-sistemas-de-recomendacion
```

Los notebooks usan datos sintéticos (`applied_recsys.synthetic`): no requieren descargas y permiten
comparar con los gustos *reales*, algo imposible con datos de verdad.

## Recursos

- C. Aggarwal, *Recommender Systems: The Textbook* (2016).
- Y. Koren, R. Bell, C. Volinsky, “Matrix Factorization Techniques for Recommender Systems”
  (*IEEE Computer*, 2009).
- Y. Hu, Y. Koren, C. Volinsky, “Collaborative Filtering for Implicit Feedback Datasets” (ICDM 2008).
- P. Cremonesi, Y. Koren, R. Turrin, “Performance of Recommender Algorithms on Top-N Recommendation
  Tasks” (RecSys 2010): PureSVD y RMSE vs. ranking.
- S. Rendle et al., “BPR: Bayesian Personalized Ranking from Implicit Feedback” (UAI 2009).
- T. Schnabel et al., “Recommendations as Treatments: Debiasing Learning and Evaluation” (ICML 2016): IPS.
- M. F. Dacrema, P. Cremonesi, D. Jannach, “Are We Really Making Much Progress?” (RecSys 2019):
  baselines bien ajustados frente a modelos neuronales.

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (3, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (5; soluciones verificadas por tests)
- [x] Mini-proyecto con README, código y resultados (diversificación: tu turno)
- [x] Tests de las funciones en `src/`
- [ ] CI en verde
