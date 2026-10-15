# Integrador S1 — Temas de libros como puente para recomendar

> Especificación de diseño · 2026-10-15 · Estado: **para revisión**
> Carpeta: `01-semestre-1/proyecto-integrador-s1/`

## 1. Objetivo

Construir, de punta a punta y con análisis estadístico serio, un sistema que:

1. **Dado un libro** (título del catálogo o una descripción libre), prediga **de qué temas trata**, con probabilidad.
2. **Dados los libros que alguien leyó**, recomiende libros **usando los temas como puente** y explique cada
   recomendación (“porque te gustan: ciencia ficción, fantasía épica”).

Decisiones tomadas con el usuario:

| Decisión | Elección |
|---|---|
| Propósito | Temas como puente (ambas mitades conectadas) |
| Idioma | Datos en inglés; código y documentación en español, como el resto del repo |
| Entregable | CLI + informe (README y notebooks). API/despliegue quedan para MLOps (S2) |
| Enfoque | A: temas curados + clasificador de texto + recomendador híbrido |

**Criterio de éxito:** las tres hipótesis de la sección 6 evaluadas con su intervalo y decisión (sea cual sea
el resultado), la CLI funcionando sobre el catálogo real, y el informe explicando qué funciona, qué no y por qué.

## 2. Fuera de alcance

- API web, contenedores y despliegue → módulo de MLOps (S2).
- Modelos de lenguaje, embeddings o transformers → módulo de PLN (S2). El integrador deja planteada la
  pregunta “¿cuánto mejora un transformer al TF-IDF?” para ese módulo.
- Libros en español.

## 3. Datos

**Fuente:** goodbooks-10k (Z. Zając, https://github.com/zygmuntz/goodbooks-10k), licencia **CC BY-SA 4.0**
(archivo `LICENSE` del repositorio). Atribución en el README del proyecto. Los datos no se suben al repo:
`data/download.py` los descarga con verificación SHA-256.

| Archivo | Contenido | Uso |
|---|---|---|
| `ratings.csv` | ~6 M calificaciones (1–5, según el README del dataset) de 53 424 lectores sobre 10 000 libros, ordenadas por tiempo | recomendación |
| `book_tags.csv` + `tags.csv` | 999 912 pares libro–estante con conteo; 34 252 estantes distintos | temas |
| `books.csv` | metadatos: autores, año, idioma, títulos, ISBN | particiones, CLI |
| `books_xml/books_xml.zip` | XML de Goodreads por libro; **9 623 / 10 000 con descripción de ≥ 20 palabras** (mediana 137) | texto |
| `to_read.csv` | ~1 M marcas “quiero leer” | no se usa en la versión base |

Hechos verificados en la exploración: 87 % de los libros en inglés (`eng`/`en-*`), 11 % sin código de idioma;
de los 60 estantes más frecuentes, ~12 son temáticos y el resto organizativos (`to-read`, `owned`, `kindle`,
`read-in-2015`…).

**Lectura del XML:** con `xml.etree.ElementTree`, no expresiones regulares. Las descripciones traen HTML:
se limpia con `html.unescape` y eliminación de etiquetas. Los 377 libros con descripción de menos de 20 palabras
(315 de ellos vacía) quedan fuera de la parte de texto pero dentro de la de recomendación.

## 4. Taxonomía de temas

### 4.1 Separar estantes temáticos de organizativos

Candidatos: los estantes usados en ≥ 100 libros (1 109). Para cada estante se calcula la **concentración**:
qué tan desproporcionadamente se reparte respecto de la popularidad de los libros. Un estante organizativo
(`to-read`) aparece en casi todos los libros en proporción a su popularidad; uno temático (`fantasy`) se
concentra en un subconjunto. Medida: coeficiente de Gini de la proporción `conteo del estante / total de
estantes del libro` a través de los libros. Se ordenan los estantes por esa medida; los extremos se aceptan
o descartan automáticamente y **solo la zona intermedia se revisa a mano**. Los umbrales elegidos y cuántos
estantes cayeron en cada grupo se reportan.

### 4.2 Agrupar sinónimos en temas

Archivo versionado `temas.yaml` con ~35–40 temas, cada uno con su nombre en español y la lista de estantes que
lo componen:

```yaml
fantasia:
  nombre: Fantasía
  estantes: [fantasy, epic-fantasy, high-fantasy, fantasy-sci-fi, ...]
```

Es el único paso con criterio humano y queda auditable en el repo. Un test verifica que ningún estante
pertenezca a dos temas y que todos los estantes listados existan en los datos.

### 4.3 Etiquetar cada libro

Un tema se asigna a un libro si la suma de los conteos de sus estantes en ese tema representa al menos una
proporción `τ` del total de conteos temáticos del libro. `τ` se elige mirando la distribución de temas por libro
(objetivo orientativo: mediana de 2–4 temas) y se reporta. Libros sin ningún tema tras el umbral se excluyen de
la parte de clasificación.

### 4.4 Validación de etiquetas

En una muestra aleatoria de 300 libros se obtienen los *subjects* de Open Library (API de búsqueda por ISBN;
datos abiertos). Para los ~10 temas con equivalente claro (fantasía, ciencia ficción, romance, misterio,
historia, biografía, autoayuda, terror, poesía, juvenil) se mapean palabras clave de los subjects a temas y se
calcula **κ de Cohen** por tema. Las consultas se cachean en `data/` y se hacen con pausas entre llamadas.
Interpretación: κ alto → las etiquetas son confiables; κ bajo → las métricas del clasificador se leen con esa
cautela, y el informe lo dice.

## 5. Modelos

Código en un paquete nuevo `book_topics` (`proyecto-integrador-s1/src/book_topics/`), registrado en
`pyproject.toml`. Reutiliza `applied_recsys` (ALS implícito, PureSVD, métricas de ranking) y `applied_ml`
(utilidades de validación) en vez de reimplementarlos.

### 5.1 Libro → temas (clasificación multi-etiqueta)

**Partición agrupada por autor principal** (`GroupShuffleSplit`): 70 % train / 10 % validación / 20 % test, con
todos los libros de un autor del mismo lado. Evita que el modelo acierte por reconocer una serie o un autor
(*Harry Potter* #1 en train y #2 en test).

| Modelo | Detalle | Rol |
|---|---|---|
| Prevalencia | predice cada tema con su frecuencia en train | piso |
| k vecinos | TF-IDF (título + descripción), coseno; temas = voto ponderado de los k más parecidos | baseline explicable |
| **Principal** | TF-IDF (1–2 gramas, sublineal) + regresión logística uno-contra-todos, probabilidades calibradas | modelo de la CLI |
| Lectores → temas | factores de ALS implícito de cada libro (de las calificaciones) + logística | pregunta exploratoria |

Hiperparámetros (C, k, dimensión de factores) y **umbral de decisión por tema** se eligen en validación
maximizando F1 por tema. El test se usa una sola vez.

### 5.2 Temas → libros (recomendación)

**Partición por lector:** el último 20 % de las calificaciones de cada lector (el archivo está ordenado por
tiempo) va a test; dentro del train, el último 10 % es validación. Relevantes = libros de test con calificación ≥ 4.

| Modelo | Detalle |
|---|---|
| Popularidad | número de calificaciones en train |
| Colaborativo | ALS implícito y PureSVD de `applied_recsys`; se queda el mejor en validación |
| **Por temas** | perfil del lector `p_u` = promedio de los vectores de temas de los libros que calificó, ponderado por (calificación − media del lector); puntaje de un libro = similitud coseno entre `p_u` y sus temas |
| **Híbrido** | `α · z(colaborativo) + (1 − α) · z(temas)`, con puntajes estandarizados por lector y `α` elegido en validación |

**Explicación:** para cada libro recomendado por temas o híbrido, los 2–3 temas con mayor contribución
(`p_u[t] · temas_i[t]`) al puntaje.

### 5.3 Experimento de libros nuevos

Se retira un 10 % de libros al azar, con sus calificaciones y estantes, del entrenamiento de *todo*. Para esos
libros solo se usa su descripción: el clasificador 5.1 predice sus temas y el modelo por temas 5.2 los puntúa
para cada lector. Evaluación: para los lectores que calificaron con ≥ 4 algún libro nuevo en su test, rankear
solo los libros nuevos. Comparación con popularidad (calculada sobre las calificaciones de test, como
referencia optimista) y con un orden aleatorio. El colaborativo no participa: no puede puntuar libros sin lectores.

## 6. Evaluación y estadística

### 6.1 Hipótesis (fijadas antes de ver resultados)

| | Hipótesis | Métrica | Prueba |
|---|---|---|---|
| **H1** | La logística supera a k vecinos | F1-macro en test | diferencia pareada por libro, IC 95 % por bootstrap de conglomerados (autores) |
| **H2** | El híbrido supera al colaborativo para los lectores del cuartil inferior de historial | NDCG@10 | diferencia pareada por lector, IC 95 % por bootstrap (lectores) |
| **H3** | Con libros nuevos, temas predichos supera a popularidad | NDCG@10 sobre libros nuevos | diferencia pareada por lector, IC 95 % por bootstrap |
| Exploratoria | ¿Descripción o lectores predicen mejor los temas? | F1-macro | IC de la diferencia, sin hipótesis direccional |

Los p-valores de H1–H3 (bootstrap, bilaterales) se corrigen con **Holm** (`applied_stats.holm`). La exploratoria
se reporta sin corrección y marcada como tal.

### 6.2 Métricas

- **Libro → temas:** F1 micro y macro, F1 por tema, precisión@3 (de los 3 temas más probables, cuántos son
  correctos), calibración (ECE y diagrama de confiabilidad).
- **Temas → libros:** NDCG@10, recall@10, cobertura del catálogo, percentil de popularidad de lo recomendado.
- **Bootstrap por conglomerados:** se remuestrean autores (parte A) o lectores (parte B), no observaciones
  sueltas, porque las observaciones dentro de un conglomerado están correlacionadas.

### 6.3 Análisis de errores (parte A)

1. F1 por tema frente a su prevalencia: ¿los temas raros fallan más?
2. Pares de temas que más se confunden (falsos positivos de un tema que son verdaderos de otro).
3. Efecto del largo de la descripción: F1 por libro según cuartil de número de palabras.

### 6.4 Advertencia conocida

La evaluación offline premia predecir lo que se observará, no lo que gusta (módulo de sistemas de
recomendación, notebook 03). El informe lo recuerda al interpretar H2 y H3.

## 7. CLI

`uv run python -m book_topics <comando>`:

| Comando | Entrada | Salida |
|---|---|---|
| `entrenar` | — | entrena y guarda los modelos en `data/modelos/` (ignorado por git) |
| `temas "<texto>"` | título del catálogo o descripción libre | temas con probabilidad, ordenados |
| `libro "<título>"` | título (búsqueda tolerante: minúsculas, sin puntuación, coincidencia parcial con `difflib`) | temas reales y predichos, y 10 libros parecidos por temas |
| `recomendar --lei T1 T2 ... [--no-me-gusto T ...]` | títulos leídos | 10 libros con explicación por temas |

`recomendar` arma el perfil de temas con los títulos dados (peso +1 por “leí”, −1 por “no me gustó”): es el caso
de un lector nuevo, sin historial en el sistema, que el modelo por temas sí puede atender.

**Errores:** si faltan modelos, mensaje indicando `entrenar`; si un título no se encuentra, mensaje con las 3
coincidencias más cercanas; si un título es ambiguo, se usa la de mayor similitud y se informa cuál.

## 8. Estructura de archivos

```
01-semestre-1/proyecto-integrador-s1/
├── README.md                 # informe: pregunta, datos, hipótesis, resultados, lectura, limitaciones
├── temas.yaml                # taxonomía curada
├── data/download.py          # goodbooks-10k con checksum (+ caché de Open Library)
├── src/book_topics/
│   ├── data.py               # carga de CSV y XML, limpieza de descripciones
│   ├── taxonomy.py           # concentración de estantes, carga de temas.yaml, etiquetado
│   ├── splits.py             # partición agrupada por autor; partición por lector
│   ├── classify.py           # baselines, logística, umbrales por tema
│   ├── recommend.py          # perfil de temas, puntaje por temas, híbrido, explicaciones
│   ├── evaluation.py         # métricas multi-etiqueta, bootstrap por conglomerados
│   └── cli.py / __main__.py  # comandos
├── notebooks/
│   ├── 01-taxonomia-y-etiquetas.ipynb
│   ├── 02-libro-a-temas.ipynb
│   └── 03-recomendacion-y-libros-nuevos.ipynb
├── tests/                    # con un catálogo diminuto en el propio test (sin datos reales)
└── resultados/               # metrics.json y figuras generadas por los scripts
```

Los notebooks requieren los datos descargados. **No se marcan `smoke`**: el CI no tiene los datos y la
corrección del código se cubre con los tests sobre catálogo sintético. Esto es una excepción explícita a la
convención de los módulos, justificada porque el integrador trabaja con datos reales por definición.

## 9. Pruebas

- Taxonomía: ningún estante en dos temas; todos los estantes de `temas.yaml` existen (test sobre los datos
  marcado para saltarse si no están descargados); el etiquetado respeta `τ` en un ejemplo a mano.
- Particiones: ningún autor en dos particiones; ningún lector pierde todo su train.
- Bootstrap por conglomerados: con datos sintéticos de correlación intra-grupo conocida, su IC es más ancho que
  el del bootstrap ingenuo y cubre el valor real.
- Recomendación: el perfil de temas y la explicación coinciden con un cálculo a mano en un catálogo de 5 libros.
- CLI: los tres comandos sobre un catálogo diminuto; mensajes de error (sin modelos, título inexistente).

## 10. Riesgos

| Riesgo | Mitigación |
|---|---|
| Estantes como verdad ruidosa | validación con Open Library (κ) y cautela explícita en el informe |
| Temas muy desbalanceados | F1-macro además de micro; umbral por tema; análisis por prevalencia |
| Tamaño: 6 M calificaciones | matrices dispersas; ALS ya probado en MovieLens 1M; submuestreo de lectores para la evaluación si hace falta, reportado |
| Open Library no disponible o con límite de tasa | caché local; si falla, la validación se reporta como no realizada (no bloquea el resto) |
