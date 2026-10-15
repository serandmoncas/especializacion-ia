# Procesamiento de lenguaje natural

> S2 · 2 créditos · Propia · **Estado: plantilla, sin desarrollar**

## Objetivos

_Por definir al desarrollar el módulo (ver `01-semestre-1/01-matematicas-aplicadas`)._

## Temas

- Tokenización (BPE, WordPiece)
- Embeddings estáticos y contextuales
- Arquitectura transformer
- Clasificación de texto

## Mini-proyecto

Clasificador de texto en español con fine-tuning. Ver [`proyecto/`](proyecto/).

## Cómo trabajar el módulo

```bash
uv sync --extra base --extra dl --extra nlp
uv run pytest 02-semestre-2/02-pln
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
