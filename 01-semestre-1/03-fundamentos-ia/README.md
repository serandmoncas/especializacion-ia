# Fundamentos de IA

> Semestre 1 · 2 créditos · Núcleo

## Objetivos

Al terminar el módulo se debe poder:

1. Describir un problema como agente + entorno (PEAS) y elegir una arquitectura de agente acorde.
2. Formular problemas de búsqueda e implementar BFS, IDDFS, UCS, Greedy, A*, WA* e IDA*.
3. Diseñar heurísticas admisibles (por relajación o precómputo) y medir su calidad.
4. Representar conocimiento en lógica proposicional y decidir implicación por modelos,
   encadenamiento y resolución.

## Temas

| Tema | Dónde |
|---|---|
| Historia y panorama, agentes, PEAS, tipos de entorno | `notebooks/01-agentes-y-panorama.ipynb` |
| Búsqueda no informada e informada, heurísticas en grillas, WA* | `notebooks/02-busqueda-en-grillas.ipynb` |
| Diseño de heurísticas, dominancia, b*, A* vs. IDA* | `notebooks/03-heuristicas-8-puzzle.ipynb` |
| Lógica proposicional, model checking, forward chaining | `notebooks/04-logica-proposicional.ipynb` |
| CSP, búsqueda adversaria, bidireccional, resolución, consistencia | `ejercicios/` |
| Pathfinding en mapas reales con heurística ALT | `proyecto/` |

## Estructura

```
03-fundamentos-ia/
├── notebooks/              # 01 agentes · 02 búsqueda · 03 heurísticas · 04 lógica
├── src/ai_foundations/
│   ├── search.py           # BFS, IDDFS, UCS, Greedy, A*, WA*, IDA*, b*
│   ├── problems/           # grid.py (reglas Moving AI), puzzle.py (8/15-puzzle)
│   ├── landmarks.py        # heurística ALT
│   ├── agents.py           # mundo de la aspiradora y tres agentes
│   ├── logic.py            # lógica proposicional
│   └── viz.py
├── tests/                  # incluye admisibilidad exhaustiva sobre los 181 440 estados del 8-puzzle
├── ejercicios/             # 4 en código + 1 en papel; soluciones/ aparte
├── data/download.py        # benchmarks Moving AI (Dragon Age: Origins), con checksum
└── proyecto/               # comparación de 6 algoritmos en 4 mapas reales
```

## Cómo trabajar el módulo

```bash
uv sync --extra base
uv run jupyter lab 01-semestre-1/03-fundamentos-ia/notebooks
uv run pytest 01-semestre-1/03-fundamentos-ia
```

## Recursos

- S. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, 4.ª ed. (2020): caps. 1–3
  (agentes, búsqueda), 5–6 (adversaria, CSP), 7 (lógica proposicional).
- A. Patel, [*Amit's A* Pages*](https://theory.stanford.edu/~amitp/GameProgramming/): heurísticas
  y detalles prácticos de pathfinding en juegos.
- A. Goldberg, C. Harrelson, “Computing the Shortest Path: A* Search Meets Graph Theory”
  (SODA 2005): el artículo de ALT.
- N. Sturtevant, “Benchmarks for Grid-Based Pathfinding”, *IEEE TCIAIG* 4(2), 2012, y
  [movingai.com/benchmarks](https://movingai.com/benchmarks/).
- R. Korf, “Depth-First Iterative-Deepening” (*AIJ*, 1985): IDDFS e IDA*.

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (4, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (5; soluciones verificadas por tests)
- [x] Mini-proyecto con README, código y resultados (extensión ALT “farthest”: pendiente)
- [x] Tests de las funciones en `src/`
- [x] CI en verde
