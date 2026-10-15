# Automatización de Operaciones en Datos e IA (MLOps)

> S2 · 2 créditos · Compartida · **Estado: plantilla, sin desarrollar**

## Objetivos

_Por definir al desarrollar el módulo (ver `01-semestre-1/01-matematicas-aplicadas`)._

## Temas

- Tracking de experimentos
- Versionado de datos y modelos
- Empaquetado y despliegue
- Monitoreo y drift

## Mini-proyecto

Servir un modelo con FastAPI + Docker + MLflow + CI. Ver [`proyecto/`](proyecto/).

## Cómo trabajar el módulo

```bash
uv sync --extra base --extra mlops
uv run pytest 02-semestre-2/04-mlops
```

## Recursos

_Por definir._

## Checklist (definición de terminado)

- [ ] README con objetivos, temas y recursos
- [ ] Al menos 2 notebooks ejecutables sin errores
- [ ] Ejercicios con soluciones en carpeta aparte
- [ ] Mini-proyecto con README, código y resultados
- [ ] Tests de las funciones en `src/`
- [ ] CI en verde
