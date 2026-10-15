# Proyecto integrador S1 · Temas de libros como puente para recomendar

> **Estado: completo** (falta CI en verde para marcarlo ✅)

Dos preguntas, un mismo puente:

- **Parte A · libro → temas.** Dado el título y la sinopsis de un libro, ¿de qué temas trata?
- **Parte B · temas → libros.** Dado lo que alguien leyó, ¿qué libros recomendarle? ¿Y qué hacer con un
  libro que **nadie ha calificado todavía**?

Los temas son el puente: se aprenden del texto (A) y sirven para recomendar incluso libros sin
historial (B). Integra estadística (bootstrap, Holm), fundamentos de IA (búsqueda de umbrales),
aprendizaje automático (clasificación multi-etiqueta calibrada) y la electiva de recomendación
(PureSVD, ALS, híbridos).

## Datos

[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) de Zygmunt Zając, licencia
**CC BY-SA 4.0**. Los datos no se versionan: `uv run python data/download.py` los baja y verifica su
SHA-256.

| | |
|---|---|
| Libros | 10 000 |
| Calificaciones (1–5) | 5 976 479 de 53 424 lectores |
| Pares libro–estante (Goodreads *shelves*) | 999 912 |
| Libros con sinopsis de ≥ 20 palabras | 9 623 |

## Diseño

**De estantes a temas** ([notebook 01](notebooks/01-taxonomia-y-etiquetas.ipynb)). Goodreads no tiene
temas; tiene estantes que cada lector inventa (`fantasy`, pero también `to-read` u `owned`).

- La spec proponía separar estantes temáticos de organizativos con el **Gini**. **No sirve**: en los
  1 109 estantes con ≥ 100 libros, el Gini tiene Spearman −0,97 con el número de libros. Mide rareza:
  `fantasy` (0,86) y `abandoned` (0,80) quedan juntos.
- Se agregó `corr_popularidad`, la correlación entre el conteo del estante y el total de estantes del
  libro. Los 20 estantes con ≥ 0,35 son todos organizativos (`to-read` 0,84, `owned` 0,57). Sin embargo,
  la zona media no se separa: `abandoned` tiene 0,30 y `fantasy` 0,23.
- Por eso el clasificador real es una **lista curada**: [`temas.yaml`](temas.yaml) tiene 40 temas y
  167 estantes. La medida ordena a los candidatos ([`candidatos_estantes.csv`](candidatos_estantes.csv))
  y una persona decide.
- Un libro tiene un tema si los estantes de ese tema suman ≥ τ = 0,05 de sus conteos temáticos. τ es el
  menor valor con una mediana de 2 a 4 temas por libro y todos los temas con ≥ 50 libros. Con él, la
  mediana es 3 temas por libro y ningún libro queda sin tema.

**Particiones**

- **Parte A:** se parte por **autor** (70/10/20; 6 690 / 810 / 2 123 libros). Si el mismo autor quedara
  en train y en test, el modelo reconocería la serie y no el contenido.
- **Parte B:** se parte por el **orden de cada lector**: su último 20 % va a test.
- **Libros nuevos:** el 10 % de los libros (960) se retira de todo el entrenamiento.

**Modelos**

| Parte | Modelo | Qué hace |
|---|---|---|
| A | Prevalencia | Predice la frecuencia de cada tema (piso). |
| A | k vecinos | Promedia los temas de los k libros con sinopsis más parecidas (coseno TF-IDF). |
| A | **Logística** | TF-IDF de 1 y 2 gramas, una logística por tema, calibración sigmoide y un umbral por tema elegido en validación. |
| A | Lectores (exploratorio) | Logística sobre el vector ALS de cada libro: lo que dicen *quienes lo leen*, no el texto. |
| B | Popularidad | Lo más calificado que el lector no ha leído. |
| B | Colaborativo | Se elige en validación entre ALS implícito y PureSVD; ganó PureSVD con k = 100. |
| B | Temas | Coseno entre el perfil de temas del lector (sus calificaciones centradas en su media, así un 3 de quien da 5 a todo resta) y los temas de cada libro. |
| B | Híbrido | α · z(colaborativo) + (1 − α) · z(temas), estandarizados por lector; α = 0,75 elegido en validación. |

## Hipótesis

Se fijaron antes de ver test. Las diferencias son pareadas. El IC 95 % sale de un bootstrap por
conglomerados (autor en A, lector en B) con 2 000 réplicas, y los p-valores se corrigen con Holm. Con
2 000 réplicas, el menor p posible es 0,0005; cuando sale ese valor se reporta "< 0,0005".

| | Comparación | Δ | IC 95 % | p | p Holm | ¿Se apoya? |
|---|---|---|---|---|---|---|
| H1 | Logística − k vecinos, F1-macro (A) | +0,063 | [+0,047; +0,077] | < 0,0005 | 0,0015 | **Sí** |
| H2 | Híbrido − colaborativo, NDCG@10 en el cuartil de menos historial (B) | +0,0005 | [+0,0001; +0,0008] | 0,013 | 0,013 | Sí, pero **irrelevante** |
| H3 | Temas predichos − popularidad, NDCG@10 en libros nuevos | −0,101 | [−0,103; −0,099] | < 0,0005 | 0,0015 | **No** |

## Resultados

### Parte A · libro → temas (test: 2 123 libros de autores no vistos)

| Modelo | F1-micro | F1-macro | Precisión@3 | ECE |
|---|---|---|---|---|
| Prevalencia | 0,249 | 0,124 | 0,249 | 0,003 |
| k vecinos (k = 50) | 0,589 | 0,488 | 0,602 | 0,030 |
| **Logística (C = 16)** | 0,644 | **0,550** | **0,695** | 0,005 |
| Lectores (ALS) | **0,663** | 0,445 | 0,638 | 0,042 |

- **Texto frente a lectores:** el texto gana en F1-macro por +0,105 (IC [0,080; 0,130]) y los lectores
  ganan en F1-micro. Los lectores aciertan los temas grandes y el texto distingue mejor los pequeños.
- **La F1 por tema sigue a la prevalencia** (Spearman 0,58).
- **Confusiones principales:** aventura → fantasía (126 falsos positivos), juvenil → fantasía (77) y
  romance → fantasía (72).
- **La sinopsis importa:** el F1-micro va de 0,566 en el cuartil más corto a 0,676 en los más largos.

### Parte B · temas → libros (53 334 lectores)

| Modelo | NDCG@10 | Cobertura del catálogo | Percentil de popularidad |
|---|---|---|---|
| Popularidad | 0,0505 | 0,6 % | 99,9 |
| Colaborativo (PureSVD) | 0,0712 | 15,0 % | 96,6 |
| Temas | 0,0179 | **61,3 %** | **72,7** |
| **Híbrido** | **0,0723** | 15,5 % | 96,5 |

Recomendar solo por temas acierta poco, pero cubre 4 veces más catálogo y propone libros menos
populares. Sirve para descubrir, no para acertar. Todos los modelos mejoran con más historial; por
ejemplo, el híbrido pasa de 0,0645 en Q1 a 0,0786 en Q4.

### Libros nuevos (960 libros sin calificaciones en entrenamiento; 52 343 lectores)

| Estrategia | NDCG@10 |
|---|---|
| Aleatorio | 0,009 |
| **Temas predichos de la sinopsis** | **0,042** |
| Popularidad (*optimista*: usa calificaciones que el día 1 no existirían) | 0,143 |

Por eso H3 no se apoya. Aun así, los temas son la única señal disponible el día 1 y multiplican por
4,6 el resultado del azar.

### ¿Son confiables las etiquetas? Acuerdo con Open Library (300 libros)

| Tema | κ | Libros con el tema (nosotros / Open Library) |
|---|---|---|
| Autoayuda | 0,13 | 14 / 1 |
| Romance | 0,16 | 82 / 15 |
| Misterio | 0,18 | 62 / 13 |
| Fantasía | 0,26 | 76 / 16 |
| Juvenil | 0,38 | 64 / 28 |
| Ciencia ficción | 0,40 | 34 / 12 |
| Terror | 0,43 | 12 / 6 |
| Historia | 0,45 | 18 / 19 |
| Biografía | 0,47 | 21 / 11 |
| Poesía | 0,67 | 2 / 1 (muy pocos casos) |

El acuerdo es de débil a moderado, y buena parte se explica porque Open Library anota muchos menos
temas. Las etiquetas de lectores son ruidosas, pero no arbitrarias, y todas las métricas de la Parte A
heredan ese techo.

### Dos ejecuciones: el desempate post hoc

La primera ejecución completa está en `resultados/metrics_v1_sin_desempate.json`. En ella, el
recomendador por temas daba 0,0098 de NDCG@10, porque muchos libros empataban en puntaje y el orden
entre empatados era arbitrario. Se agregó un desempate por popularidad con un peso menor que 1e-9
(`topic_scores_with_tiebreak`) y el resultado subió a 0,0179. Los otros modelos no cambiaron.

El ajuste es razonable, pero se hizo **después de ver test**, así que se reporta como post hoc. Las
hipótesis no dependen de él: H3 usa temas *predichos*, y el desempate no se aplica a libros nuevos.

## Limitaciones

- **Los estantes no son temas.** Son la opinión de lectores de Goodreads, en su mayoría angloparlantes
  y aficionados a ciertos géneros. El κ con Open Library muestra cuánto ruido arrastran.
- **Evaluación offline.** NDCG sobre lo que el lector calificó después no mide si una recomendación le
  habría gustado. Un libro bueno que no leyó cuenta como error.
- **Solo inglés.** El modelo de texto se entrenó con sinopsis en inglés. Para textos que no lo parecen,
  la CLI muestra un aviso de baja confianza (palabras conocidas < 50 % o más palabras funcionales en
  español que en inglés). Es un aviso, no una solución.
- **TF-IDF ve palabras, no intención.** *The Alchemist* tiene 7 temas según sus lectores, pero el
  modelo solo predice "Literatura" a partir de la sinopsis. Ver el
  [notebook 02](notebooks/02-libro-a-temas.ipynb).

## Uso

```bash
uv run python data/download.py                 # ~ 140 MB
uv run python run.py                           # experimentos completos → resultados/ (~ 5 min con caché)
uv run python -m book_topics entrenar          # modelo de la CLI → data/modelos/bundle.joblib
uv run python -m book_topics temas "Dune"
uv run python -m book_topics temas "A young wizard discovers a hidden school of magic..."
uv run python -m book_topics libro "The Hobbit"
uv run python -m book_topics recomendar --lei Dune Foundation "The Hobbit"
uv run python -m book_topics recomendar --lei "Pride and Prejudice" "Jane Eyre" --no-me-gusto Twilight
```

Cada recomendación explica en qué temas se basa ("porque te gustan: Clásicos, Ciencia ficción").

Los notebooks necesitan los datos y `resultados/`, así que **no son smoke** y el CI no los ejecuta.

- [01 · De estantes a temas](notebooks/01-taxonomia-y-etiquetas.ipynb)
- [02 · Libro → temas](notebooks/02-libro-a-temas.ipynb)
- [03 · Temas → libros y libros nuevos](notebooks/03-recomendacion-y-libros-nuevos.ipynb)

> `data/modelos/bundle.joblib` usa pickle: carga solo bundles que generaste tú con `entrenar`.

## Preguntas para el semestre 2 (PLN)

- ¿Un modelo de *embeddings* de oraciones o un transformer afinado supera al TF-IDF en F1-macro? ¿Sobre
  todo en los temas raros y en sinopsis cortas, donde TF-IDF cae a 0,566?
- ¿Se cierra la brecha con los lectores en F1-micro si el texto captura intención y tono, no solo
  vocabulario?
- **Español:** con un modelo multilingüe, ¿se pueden clasificar sinopsis en español sin reentrenar, y
  cuánto se pierde frente al inglés?
- ¿Mejoran los temas predichos con mejor texto lo suficiente para acercarse a la popularidad en libros
  nuevos (H3)?
