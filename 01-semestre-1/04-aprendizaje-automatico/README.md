# Aprendizaje automático

> Semestre 1 · 2 créditos · Compartida

## Objetivos

Al terminar el módulo se debe poder:

1. Estimar el desempeño de un modelo sin engañarse: CV estratificada, CV anidada, sin fugas.
2. Implementar y entender regresión logística, árboles, random forest, boosting y k-means.
3. Elegir métricas según el problema (desbalance, costos, calibración) y fijar umbrales sin tocar el test.
4. Construir un pipeline de scikit-learn completo sobre datos tabulares reales.

## Temas

| Tema | Dónde |
|---|---|
| Sesgo–varianza, variabilidad de la CV, fugas, optimismo de selección | `notebooks/01-validacion-y-fugas.ipynb` |
| Árboles, sobreajuste, bagging, random forest, OOB, boosting | `notebooks/02-arboles-y-ensambles.ipynb` |
| Solvers (GD vs. Newton), desbalance, ROC/PR, umbrales por costo, calibración | `notebooks/03-metricas-y-calibracion.ipynb` |
| K-means, elección de k, límites, PCA + clustering | `notebooks/04-no-supervisado.ipynb` |
| k-NN y dimensión, naive Bayes, isotónica, permutación | `ejercicios/` |
| Pipeline completo sobre Adult | `proyecto/` |

## Estructura

```
04-aprendizaje-automatico/
├── notebooks/               # 01 validación · 02 árboles · 03 métricas · 04 no supervisado
├── src/applied_ml/          # API compatible con sklearn (BaseEstimator)
│   ├── linear.py            # logística (GD y Newton/IRLS)
│   ├── tree.py              # CART clasificación/regresión, búsqueda de cortes vectorizada
│   ├── ensemble.py          # random forest (con OOB), gradient boosting (hojas de Newton)
│   ├── cluster.py           # k-means + k-means++
│   ├── metrics.py           # ROC/AUC por rangos, PR/AP, log-loss, Brier, calibración
│   ├── model_selection.py   # k-fold estratificado, CV, IC bootstrap de métricas
│   └── adult.py             # carga y limpieza de UCI Adult (la reutiliza el módulo de ética)
├── tests/                   # cada estimador contra sklearn
├── ejercicios/              # 4 en código + 1 en papel; soluciones/ aparte
├── data/download.py         # UCI Adult, con checksum
└── proyecto/                # CV anidada, desde cero vs. sklearn, evaluación final, subgrupos
```

## Cómo trabajar el módulo

```bash
uv sync --extra base
uv run jupyter lab 01-semestre-1/04-aprendizaje-automatico/notebooks
uv run pytest 01-semestre-1/04-aprendizaje-automatico
```

## Recursos

- T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2.ª ed. (2009):
  caps. 4, 7 (validación, §7.10.2 fuga por selección), 9–10, 14–15. Libre en la web de los autores.
- G. James et al., *An Introduction to Statistical Learning with Python* (2023): versión aplicada.
- S. Raschka, “Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning”
  (arXiv:1811.12808, 2018): CV anidada y comparación de modelos.
- [Guía de usuario de scikit-learn](https://scikit-learn.org/stable/user_guide.html): pipelines,
  calibración, inspección.
- A. Niculescu-Mizil, R. Caruana, “Predicting Good Probabilities with Supervised Learning” (ICML 2005).

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (4, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (5; soluciones verificadas por tests)
- [x] Mini-proyecto con README, código y resultados (ingeniería de features: tu turno)
- [x] Tests de las funciones en `src/`
- [ ] CI en verde
