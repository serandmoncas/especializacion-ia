# HANDOFF — Repo de estudio: Especialización en IA (malla tipo EAFIT)

> Documento de arranque para un agente de código (Claude Code u otro) o para quien inicialice el repo.
> Objetivo: crear un repositorio que recorra, de forma práctica y con proyectos, la malla de una
> Especialización en IA de 24 créditos (2 semestres), más un preparatorio de nivelación.

---

## 1. Contexto

La malla de referencia tiene esta estructura:

| Bloque | Curso | Créditos | Tipo |
|---|---|---|---|
| Preparatorio | Fundamentos Matemáticos | — | Nivelación |
| Preparatorio | Fundamentos Estadísticos | — | Nivelación |
| Preparatorio | Fundamentos Computacionales | — | Nivelación |
| S1 | Matemáticas aplicadas | 2 | Compartida |
| S1 | Estadística aplicada | 2 | Compartida |
| S1 | Fundamentos de IA | 2 | Núcleo |
| S1 | Aprendizaje automático | 2 | Compartida |
| S1 | Aprendizaje profundo | 2 | Compartida |
| S1 | Electiva 1 | 2 | Electiva |
| S2 | Visión por computador | 2 | Propia |
| S2 | Procesamiento de lenguaje natural | 2 | Propia |
| S2 | IA Generativa | 2 | Propia |
| S2 | Automatización de Operaciones en Datos e IA (MLOps) | 2 | Compartida |
| S2 | Ética de la IA e Integridad de los Datos | 2 | Propia |
| S2 | Electiva 2 | 2 | Electiva |

Lógica: **S1 = fundamentos** (tronco común con otros posgrados de datos); **S2 = aplicación** (visión,
lenguaje, generativa, producción y ética).

Perfil del dueño del repo: ingeniero de sistemas y desarrollador Python senior (FastAPI, Next.js,
flujos agénticos). No hace falta explicar programación básica; sí conviene rigor en matemáticas y
experimentación.

---

## 2. Objetivos del repo

1. Un módulo por curso, con: resumen de conceptos, notebooks ejecutables, ejercicios y **un mini-proyecto**.
2. Un **proyecto integrador por semestre** que use lo aprendido.
3. Todo reproducible: entorno fijado, datasets descargables por script, semillas fijas.
4. Progreso visible: checklist por módulo en el README raíz.

No-objetivos: no replicar contenido con derechos de autor de EAFIT; no buscar exhaustividad académica,
sino comprensión aplicada.

---

## 3. Stack

- **Python 3.12** con **uv** para entorno y dependencias.
- **Jupyter / JupyterLab** para notebooks; **ruff** para lint/format; **pytest** para tests.
- Numérico y ML: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`.
- Deep learning: `torch`, `torchvision`.
- NLP / generativa: `transformers`, `datasets`, `sentence-transformers`; SDK de Anthropic para LLMs.
- MLOps: `mlflow` (tracking), `fastapi` + `uvicorn` (servir modelos), Docker, GitHub Actions.
- Datos de módulos: carpetas `data/` ignoradas por git; scripts `download.py` por módulo.

---

## 4. Estructura propuesta

```
especializacion-ia/
├── README.md                 # visión, mapa de la malla, checklist de progreso
├── HANDOFF.md                # este documento
├── CLAUDE.md                 # instrucciones para el agente (ver sección 7)
├── pyproject.toml
├── .github/workflows/ci.yml  # ruff + pytest + ejecutar notebooks marcados como "smoke"
├── common/                   # utilidades compartidas (semillas, plots, carga de datos)
├── 00-preparatorio/
│   ├── 01-fundamentos-matematicos/
│   ├── 02-fundamentos-estadisticos/
│   └── 03-fundamentos-computacionales/
├── 01-semestre-1/
│   ├── 01-matematicas-aplicadas/
│   ├── 02-estadistica-aplicada/
│   ├── 03-fundamentos-ia/
│   ├── 04-aprendizaje-automatico/
│   ├── 05-aprendizaje-profundo/
│   ├── 06-electiva-1/
│   └── proyecto-integrador-s1/
└── 02-semestre-2/
    ├── 01-vision-por-computador/
    ├── 02-pln/
    ├── 03-ia-generativa/
    ├── 04-mlops/
    ├── 05-etica-e-integridad-datos/
    ├── 06-electiva-2/
    └── proyecto-integrador-s2/
```

Cada módulo contiene:

```
NN-nombre/
├── README.md        # objetivos, temas, recursos, checklist
├── notebooks/
├── src/             # código reutilizable del módulo
├── tests/
├── data/            # (gitignored) + download.py
└── proyecto/        # mini-proyecto con su propio README
```

---

## 5. Contenido mínimo por módulo

| Módulo | Temas clave | Mini-proyecto sugerido |
|---|---|---|
| Matemáticas aplicadas | Álgebra lineal, cálculo multivariable, gradientes, optimización | Regresión lineal y descenso de gradiente desde cero en NumPy |
| Estadística aplicada | Probabilidad, inferencia, pruebas de hipótesis, bootstrap, bayesiana básica | A/B test sobre un dataset público |
| Fundamentos de IA | Búsqueda, agentes, representación del conocimiento, historia y panorama | Agente de búsqueda (A*) para un puzzle o mapa |
| Aprendizaje automático | Supervisado/no supervisado, validación, métricas, feature engineering | Pipeline sklearn completo con validación cruzada |
| Aprendizaje profundo | MLP, CNN, RNN, backprop, regularización | Red en PyTorch entrenada y comparada con baseline |
| Visión por computador | Convoluciones, transfer learning, detección, segmentación | Clasificador por transfer learning (p. ej. hongos u hojas) |
| PLN | Tokenización, embeddings, transformers, clasificación de texto | Clasificador de texto en español con fine-tuning |
| IA Generativa | LLMs, prompting, RAG, agentes y uso de herramientas, evaluación | Sistema RAG sobre documentos propios con evaluación |
| MLOps | Tracking, versionado, empaquetado, despliegue, monitoreo | Servir un modelo con FastAPI + Docker + MLflow + CI |
| Ética e integridad | Sesgo, equidad, privacidad, explicabilidad, calidad de datos | Auditoría de sesgo y explicabilidad de un modelo propio |

Proyectos integradores:
- **S1:** modelo predictivo end-to-end sobre datos tabulares reales, con análisis estadístico serio.
- **S2:** aplicación de IA desplegada (visión, PLN o generativa) con MLOps y una sección de ética.

---

## 6. Primeras tareas (para el agente)

1. Crear el repo con la estructura de la sección 4 (carpetas + README vacío con plantilla en cada módulo).
2. Inicializar `pyproject.toml` con uv, dependencias agrupadas por extras (`base`, `dl`, `nlp`, `mlops`).
3. Configurar ruff, pytest y el workflow de CI.
4. Escribir el `README.md` raíz con el mapa de la malla y la checklist de progreso.
5. Crear `common/` con `set_seed()`, helpers de plotting y de rutas de datos.
6. Desarrollar completo **un** módulo como plantilla de referencia: `01-semestre-1/01-matematicas-aplicadas`.
7. Parar y pedir revisión antes de generar el resto de módulos.

---

## 7. Contenido sugerido para `CLAUDE.md`

```markdown
# Instrucciones del repo
- Idioma: español para docs y comentarios; nombres de código en inglés.
- Gestor: uv. Nunca usar pip directamente.
- Antes de commitear: `uv run ruff check --fix && uv run ruff format && uv run pytest`.
- Notebooks: celdas cortas, cada notebook ejecutable de principio a fin, semillas fijas.
- No subir datos ni pesos de modelos; usar scripts de descarga.
- Cada módulo sigue la plantilla de 01-semestre-1/01-matematicas-aplicadas.
- Priorizar implementar desde cero antes de usar la librería, y luego comparar.
```

---

## 8. Definición de terminado (por módulo)

- [ ] README con objetivos, temas y recursos
- [ ] Al menos 2 notebooks ejecutables sin errores
- [ ] Ejercicios con soluciones en carpeta aparte
- [ ] Mini-proyecto con README, código y resultados
- [ ] Tests de las funciones en `src/`
- [ ] CI en verde

---

## 9. Decisiones abiertas

- Electivas 1 y 2: ¿qué temas? (opciones: series de tiempo, aprendizaje por refuerzo, IA en el borde/IoT, sistemas de recomendación).
- ¿Repo público o privado?
- ¿Uso personal o base para un curso para otras personas?
