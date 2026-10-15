# PLN (S2) · Motor de rutas de investigación — diseño

> Estado: aprobado por secciones en conversación el 2026-10-15; pendiente de revisión escrita.

## 1. Propósito

Módulo S2 «Procesamiento de lenguaje natural» con un mini-proyecto que el usuario usaría: un **motor
que, dado un tema, dice por dónde empezar a investigar**. «Meta aprendizaje» en el sentido de
*aprender a aprender* (orientar una investigación), no de *meta-learning* en ML.

Salida para una consulta como «aprendizaje por refuerzo»:

1. **Ruta de conceptos**: qué entender antes de qué (prerrequisitos), tomada de Wikipedia en español.
2. **Lecturas por concepto**: 2–3 trabajos reales (OpenAlex), introductorios primero, cada uno con
   el porqué (revisión, citado por X % del tema, etc.).

Lo dijo el usuario: continuar la línea del integrador S1, un motor «según un tema, por dónde
comenzar», corpus mixto (conceptos + lecturas), enfoque A (grafo + clasificadores evaluados por
capas), tiene claves de Anthropic y OpenAlex. Supuestos míos: el resto de este documento.

## 2. Datos

### 2.1 AL-CPL (prerrequisitos entre conceptos)

Liang et al., «Investigating Active Learning for Concept Prerequisite Learning» (EAAI 2018).
Repo `github.com/harrylclc/AL-CPL-dataset`, carpeta `data/`. Construido sobre el *Wiki concept map*
de Wang et al. (2016), sacado de libros de texto y corregido por expertos.

| Dominio | Conceptos | Pares | Prerrequisitos |
|---|---|---|---|
| Data mining | 120 | 826 | 292 |
| Geometry | 89 | 1 681 | 524 |
| Physics | 153 | 1 962 | 487 |
| Precalculus | 224 | 2 060 | 699 |
| **Total** | 586 | 6 529 | 2 002 |

- Formato: `<dominio>.pairs` (todos los pares) y `<dominio>.preqs` (positivos), CSV sin encabezado,
  títulos de la Wikipedia en inglés. En (A, B), **B es prerrequisito de A**.
- **Expansión deliberada de los autores:** si (A, B) es positivo se agregó (B, A) como negativo, y se
  agregó el cierre transitivo. Una partición aleatoria de pares filtra la respuesta (el par invertido
  está en train). → partición por dominio (§4.1).
- `download.py` fija el commit del repo y el SHA-256 de cada archivo.

### 2.2 Wikipedia (es y en)

API de MediaWiki, sin clave. Por concepto: resumen inicial (`extracts`, solo la introducción),
enlaces salientes de la página (`links`, espacio de nombres 0), equivalencia entre idiomas
(`langlinks`) y búsqueda de títulos (para sugerencias). Se sigue `continue` hasta agotar.

- Mapeo en→es por `langlinks`, resolviendo redirecciones. Se reporta la **cobertura**: conceptos sin
  página ES y pares que se pierden. La capa 1 se evalúa sobre los pares con los dos conceptos en ES.
- Enlaces entrantes (necesarios para RefD) solo dentro del conjunto de conceptos del dominio: se
  derivan de los salientes de cada concepto, no de `backlinks` (que devuelve toda Wikipedia).

### 2.3 OpenAlex (lecturas)

API `api.openalex.org`, clave gratuita en `OPENALEX_API_KEY` (1 USD/día; sin clave, 0,10 USD/día
compartido por IP). Campos: `id`, `title`, `type`, `publication_year`, `language`, `cited_by_count`,
`referenced_works`, `abstract_inverted_index` (se reconstruye a texto). Paginación por cursor.

### 2.4 Caché

Toda respuesta HTTP exitosa se guarda en `data/cache/<fuente>/<sha1 de la petición>.json` (ignorado
por git). Las respuestas de error nunca se guardan. `run.py` reporta llamadas hechas y aciertos de
caché. Las respuestas del LLM también se guardan (clave: modelo + prompt), así se pagan una vez.

## 3. Arquitectura

Módulo `02-semestre-2/02-pln`, paquete `src/research_paths/` (registrado en `pyproject.toml`).

| Archivo | Responsabilidad |
|---|---|
| `tokenize.py` | BPE desde cero (entrenar, codificar, decodificar) |
| `attention.py` | atención escalada y multi-cabeza en NumPy |
| `retrieval.py` | BM25 desde cero; búsqueda densa (coseno sobre embeddings) |
| `http.py` | GET con caché en disco, reintentos con espera exponencial ante 429/5xx, tope de intentos |
| `wiki.py` | cliente de Wikipedia (resúmenes, enlaces, langlinks, búsqueda) |
| `openalex.py` | cliente de OpenAlex; resumen invertido → texto |
| `alcpl.py` | carga de AL-CPL, mapeo en→es, particiones por dominio |
| `prereq/refd.py` | RefD desde cero |
| `prereq/models.py` | embeddings + logística; cross-encoder XLM-R; juez LLM |
| `concepts.py` | candidatos de concepto para una consulta y orden de la ruta |
| `readings.py` | candidatos de lectura, clasificador introductorio, `start_here_score` |
| `evaluation.py` | F1, PR-AUC, recall@k, NDCG@k, bootstrap por conglomerados, Holm |
| `cli.py` / `__main__.py` | `uv run python -m research_paths ruta "tema"` |

Flujo de la CLI: consulta → página del tema en Wikipedia ES (o sugerencias si no existe) →
candidatos (conceptos enlazados desde el tema, ordenados por enlaces mutuos y similitud de embeddings,
máximo 12) → prerrequisitos por pares con el cross-encoder guardado → orden por número de victorias,
con desempate topológico (`order_route`) → por concepto, candidatos de OpenAlex (es y en) → orden por
`start_here_score` → texto con el porqué de cada elemento.

`proyecto/run.py` hace los experimentos completos y escribe `proyecto/resultados/metrics.json` y
figuras. Los modelos van a `data/modelos/` (ignorado por git).

Modelos: `intfloat/multilingual-e5-base` (embeddings), `xlm-roberta-base` (cross-encoder y
clasificador introductorio), `claude-haiku-5-5` (juez LLM). Entrenamiento en MPS si está disponible.

## 4. Evaluación

### 4.1 Capa 1 · prerrequisitos

- **Partición:** dejar un dominio fuera (4 rondas). Hiperparámetros y umbrales se eligen con los 3
  dominios de entrenamiento (validación: uno de ellos rotando). Las predicciones de las 4 rondas se
  juntan para las métricas globales.
- **Texto de un concepto:** título + resumen inicial de Wikipedia, truncado a 256 tokens por concepto.
- **Métricas:** F1 de la clase positiva y PR-AUC, globales y por dominio.

| Modelo | Detalle |
|---|---|
| Prevalencia | piso |
| RefD | Liang et al. 2015. RefD(A,B) = Σ_c w(c,A)·r(c,B) / Σ_c w(c,A) − Σ_c w(c,B)·r(c,A) / Σ_c w(c,B); w(c,X) = 1 si X enlaza a c, r(c,Y) = 1 si c enlaza a Y o c = Y. RefD(A,B) > θ ⇒ B es prerrequisito de A; θ se elige en entrenamiento. Enlaces de Wikipedia ES |
| Embeddings + logística | e5 multilingüe; rasgos [a, b, a−b, a·b] |
| Cross-encoder | XLM-R base afinado sobre «texto A </s> texto B» |
| Juez LLM | Claude Haiku, zero-shot, mismos resúmenes en ES, salida estructurada (sí/no + razón). Respuesta no interpretable = «no»; se reporta el conteo |
| Cross-encoder en inglés | igual, con resúmenes de la Wikipedia en inglés, sobre los mismos pares |

### 4.2 Capa 2 · lecturas

- **Conceptos evaluados:** los de AL-CPL con página ES que tengan una revisión en OpenAlex
  (`type:review`, título que contiene el nombre del concepto en inglés o español, ≥ 20 referencias,
  publicada 2010–2024). Se toma la más citada. Se reporta cuántos conceptos califican.
- **Referencia:** los `referenced_works` de esa revisión.
- **Candidatos:** búsqueda del concepto en OpenAlex (en y es), máximo 200, **publicados antes del
  año de la revisión**, sin la revisión misma.
- **Métricas:** recall@20 y NDCG@20. Se reporta también el **techo**: fracción de referencias que
  aparece entre los candidatos.
- **Órdenes comparados:** relevancia por defecto de OpenAlex; más citados; BM25; denso (e5);
  **combinado** (`start_here_score`: suma con pesos iguales de relevancia densa, probabilidad
  «introductorio» y log(1 + citas), cada una estandarizada dentro de los candidatos del concepto;
  fijado antes de ver test).
- **Clasificador introductorio:** XLM-R sobre resúmenes; etiqueta débil `review` (1) frente a
  `article` (0) en OpenAlex, balanceada, de los temas de los 4 dominios, **excluyendo las revisiones
  de referencia y sus referencias**. Se valida además con 100 resúmenes etiquetados por Claude durante la implementación con un
  criterio escrito en el README («¿sirve para empezar?»), auditable por el usuario.

### 4.3 Hipótesis (antes de ver test)

Bootstrap por conglomerados con 2 000 réplicas: en H1–H3 el conglomerado es el concepto objetivo
(primera columna del par); en H4, el concepto evaluado. IC 95 %, corrección de Holm sobre las cuatro.
«Se apoya» = se rechaza H0 **y** la diferencia va en la dirección prevista.

| | Hipótesis |
|---|---|
| H1 | F1(cross-encoder) − F1(RefD) > 0 en dominios no vistos |
| H2 | F1(juez LLM) − F1(cross-encoder) > 0 |
| H3 | F1(cross-encoder en inglés) − F1(cross-encoder en español) > 0, mismos pares |
| H4 | recall@20(combinado) − recall@20(más citados) > 0 |

### 4.4 Demo

Rutas para «aprendizaje por refuerzo», «series de tiempo» y «moniliasis del cacao» (temas del propio
repo) en el README del proyecto, como ejemplos cualitativos, no como métrica.

## 5. Errores

- Claves solo desde el entorno; nunca en código, caché ni logs. Si falta una, el error dice cuál y
  dónde obtenerla. `run.py --sin-llm` sigue sin Anthropic y marca H2 «no evaluada».
- 429/5xx: reintento con espera exponencial (máx. 5 intentos) y luego error claro.
- Tema sin página en Wikipedia ES: la CLI sugiere los 5 títulos más parecidos y termina con código 1.
- Concepto sin lecturas: «sin lecturas encontradas», sin relleno.
- Modelos no entrenados: la CLI dice qué comando correr.

## 6. Módulo (plantilla del repo)

- **Notebooks smoke** (sin red, segundos, datos sintéticos): 01 tokenización (BPE propio vs.
  `tokenizers`; cómo parte XLM-R palabras en español), 02 embeddings (estáticos vs. contextuales;
  espacio compartido es↔en), 03 atención y transformer (NumPy vs. `torch`), 04 clasificación afinada
  (un transformer diminuto en un corpus de juguete), 05 recuperación (BM25 vs. denso).
  Los que necesiten modelos preentrenados usan uno diminuto aleatorio o se marcan no-smoke.
- **Ejercicios** (stub + `check()`, soluciones aparte): WordPiece, máscara causal, codificación
  posicional sinusoidal, TF-IDF desde cero, MRR/NDCG, RefD en un grafo pequeño.
- **Tests** (CI solo con `base`, sin red): BPE vs. `tokenizers`; atención vs.
  `scaled_dot_product_attention` (importorskip); BM25 vs. valores calculados a mano; RefD en un
  ejemplo a mano; particiones sin conceptos compartidos y sin pares invertidos entre train y test;
  corte temporal; clientes HTTP con respuestas grabadas (incluye 429, paginación y resumen
  invertido); smoke de `run.py` en un mini-dominio sintético y de la CLI con caché grabada.
- **Dependencias:** extra `nlp` existente (transformers, datasets, sentence-transformers, anthropic);
  se agrega `tokenizers` si no viene ya con transformers.

## 7. Fuera de alcance

Interfaz web; dumps completos de Wikipedia; prerrequisitos entre lecturas; personalización por lo que
el usuario ya sabe; idiomas de consulta distintos de es/en.

## 8. Contribución del usuario

`order_route(concepts, prereq_proba)` en `concepts.py`: cómo convertir probabilidades de
prerrequisito por pares (que pueden ser inconsistentes: A→B, B→C y C→A) en una ruta lineal. Opciones:
contar victorias, orden topológico tras romper ciclos por la arista más débil, o Borda ponderado. Se
entrega con la versión mínima (contar victorias) y un TODO. No afecta ninguna hipótesis (la capa 1
se evalúa por pares); solo cambia la CLI y la demo, y el README lo dice.

## 9. Criterios de terminado

Checklist del módulo completa; `run.py` reproduce `metrics.json` con la caché llena en < 60 min en
un M4; cada cifra del README sale de `metrics.json`; CI en verde.
