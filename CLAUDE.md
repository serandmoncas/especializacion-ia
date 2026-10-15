# Instrucciones del repo

- Idioma: español para docs y comentarios; nombres de código en inglés.
- Gestor: uv. Nunca usar pip directamente.
- Antes de commitear: `uv run ruff check --fix && uv run ruff format && uv run pytest`.
- Notebooks: celdas cortas, cada notebook ejecutable de principio a fin, semillas fijas.
- No subir datos ni pesos de modelos; usar scripts de descarga.
- Cada módulo sigue la plantilla de 01-semestre-1/01-matematicas-aplicadas.
- Priorizar implementar desde cero antes de usar la librería, y luego comparar.

## Convenciones técnicas

- **Código de un módulo**: paquete en `src/<nombre_en_ingles>/`, registrado en
  `[tool.hatch.build.targets.wheel].packages` de `pyproject.toml`; luego `uv sync`.
  Nunca usar `sys.path.insert`.
- **Aleatoriedad**: `rng = common.set_seed(42)` y pasar `rng` explícito; evitar `np.random.*` global.
- **Rutas de datos**: `common.data_dir("01-semestre-1/NN-modulo")`, no rutas relativas al cwd.
- **Notebooks smoke**: marcar con `"smoke": true` en la metadata del notebook solo si corren en
  segundos y sin descargar datos; CI los ejecuta con `scripts/run_notebooks.py`.
  Guardar salidas: `uv run python scripts/run_notebooks.py --inplace <ruta>`.
- **Ejercicios**: stub en `ejercicios/ejNN_*.py` con `check()`; solución completa en
  `ejercicios/soluciones/` con el mismo `check()`; `tests/test_soluciones.py` las ejecuta.
- **Dependencias pesadas** (torch, transformers): en tests usar `pytest.importorskip`, porque
  CI solo instala el extra `base`.
- Al terminar un módulo, actualizar la tabla de progreso del README raíz.
