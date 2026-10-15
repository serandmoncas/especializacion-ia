# Sistemas de recomendación

> Electiva 1 (S1) · **Estado: plantilla, sin desarrollar**

## Temas

- Filtrado colaborativo basado en vecinos
- Factorización de matrices (ALS, SGD)
- Retroalimentación implícita y métricas de ranking
- Arranque en frío y modelos de contenido

## Mini-proyecto

Recomendador de películas evaluado con métricas de ranking. Ver [`proyecto/`](proyecto/).

## Cómo trabajar el módulo

```bash
uv sync --extra base
uv run pytest 01-semestre-1/06-electiva-1/b-sistemas-de-recomendacion
```

## Checklist (definición de terminado)

- [ ] README con objetivos, temas y recursos
- [ ] Al menos 2 notebooks ejecutables sin errores
- [ ] Ejercicios con soluciones en carpeta aparte
- [ ] Mini-proyecto con README, código y resultados
- [ ] Tests de las funciones en `src/`
- [ ] CI en verde
