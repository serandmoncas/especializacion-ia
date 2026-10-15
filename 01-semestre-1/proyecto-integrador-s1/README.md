# Proyecto integrador S1 · Temas de libros como puente para recomendar

> **Estado: completo** ✅

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
- **Libros nuevos:** el 10 % de los libros (1 000) se retira de las calificaciones de la Parte B y del
  modelo de texto que los clasifica; se evalúan los 960 que tienen sinopsis. En la Parte A siguen
  siendo libros normales del catálogo.

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
| H2 | Híbrido − colaborativo, NDCG@10 en el cuartil de menos historial (B) | +0,0008 | [+0,0005; +0,0012] | < 0,0005 | 0,0015 | Sí, pero **irrelevante** |
| H3 | Temas predichos − popularidad, NDCG@10 en libros nuevos | −0,100 | [−0,102; −0,098] | < 0,0005 | 0,0015 | **No** |

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
- **La ECE no prueba buena calibración:** agregada sobre 40 temas la dominan los negativos fáciles, y
  hasta la prevalencia saca 0,003. Habría que mirar la confiabilidad por tema.
- **La F1 por tema sigue a la prevalencia** (Spearman 0,58).
- **Errores más frecuentes:** a libros de fantasía, el modelo les agrega temas que no tienen: aventura
  (126 veces), juvenil (77) y romance (72).
- **La sinopsis importa:** el F1-micro va de 0,566 en el cuartil más corto a 0,676 en los más largos.

### Parte B · temas → libros (53 287 lectores)

| Modelo | NDCG@10 | Cobertura del catálogo | Percentil de popularidad |
|---|---|---|---|
| Popularidad | 0,0377 | 0,6 % | 99,9 |
| Colaborativo (PureSVD) | 0,0539 | 15,5 % | 96,5 |
| Temas | 0,0134 | **61,7 %** | **72,7** |
| **Híbrido** | **0,0548** | 16,0 % | 96,5 |

Recomendar solo por temas acierta poco, pero cubre 4 veces más catálogo y propone libros menos
populares. Sirve para descubrir, no para acertar. Todos los modelos mejoran con más historial; por
ejemplo, el híbrido pasa de 0,0494 en Q1 a 0,0601 en Q4.

### Libros nuevos (960 libros sin calificaciones en entrenamiento; 52 377 lectores)

| Estrategia | NDCG@10 |
|---|---|
| Aleatorio | 0,009 |
| **Temas predichos de la sinopsis** | **0,043** |
| Popularidad (*optimista*: usa calificaciones que el día 1 no existirían) | 0,143 |

Por eso H3 no se apoya. Aun así, los temas son la única señal disponible el día 1 y multiplican por
4,7 el resultado del azar.

### ¿Son confiables las etiquetas? Acuerdo con Open Library (300 libros)

| Tema | κ | Libros con el tema (nosotros / Open Library) |
|---|---|---|
| Misterio | 0,02 | 52 / 3 |
| Romance | 0,09 | 66 / 8 |
| Autoayuda | 0,12 | 13 / 2 |
| Juvenil | 0,14 | 54 / 6 |
| Fantasía | 0,15 | 76 / 12 |
| Terror | 0,19 | 18 / 2 |
| Ciencia ficción | 0,20 | 39 / 5 |
| Historia | 0,23 | 29 / 14 |
| Biografía | 0,28 | 27 / 6 |
| Poesía | 0,33 | 5 / 1 (muy pocos casos) |

El acuerdo es débil, y buena parte se explica porque Open Library anota muchos menos temas (misterio:
52 libros nuestros frente a 3 suyos). Además, **κ es inestable con 300 libros**: una muestra anterior,
tomada antes de fijar la semilla de esta validación, dio entre 0,13 y 0,47 (historia 0,45, biografía
0,47). Las etiquetas de lectores son ruidosas pero no arbitrarias, y todas las métricas de la Parte A
heredan ese techo; medir el acuerdo en serio pediría una muestra mayor o etiquetado manual.

### El desempate post hoc

La primera ejecución completa está en `resultados/metrics_v1_sin_desempate.json`. En ella, el
recomendador por temas daba 0,0098 de NDCG@10, porque muchos libros empataban en puntaje y el orden
entre empatados era arbitrario. Se agregó un desempate por popularidad con un peso menor que 1e-9
(`topic_scores_with_tiebreak`) y el resultado subió a 0,0179. Ambas cifras se midieron con la partición
por lector anterior a la corrección descrita abajo, así que no se comparan con la tabla de arriba.

El ajuste es razonable, pero se hizo **después de ver test**, así que se reporta como post hoc. H3 no
depende de él (usa temas *predichos* y el desempate no se aplica a libros nuevos), pero **H2 sí**: el
híbrido usa el puntaje de temas con desempate. En la v1, H2 daba Δ = +0,00047 con p = 0,012; con el
desempate, +0,00046 con p = 0,013. La conclusión no cambia, pero H2 ya no es estrictamente prerregistrada.

### Correcciones tras la revisión final

- **Partición por lector.** `order` se calculaba antes de quitar los libros nuevos, así que quedaba con
  huecos y test recibía ≈ 27 % de las calificaciones de cada lector en vez de 20 %. No había fuga (el
  orden temporal se respetaba), pero las cifras de la Parte B se recalcularon: todas bajan (≈ 0,071 →
  0,054 en el colaborativo) porque hay menos libros relevantes por lector, y las conclusiones se
  mantienen.
- **Modelo de la CLI.** Filtraba los libros con otro criterio que `run.py`, y 1 651 de los 2 123 libros
  de test estaban en su entrenamiento. Ahora ambos usan `usable_books` y el modelo guarda sus `test_ids`.
  Las métricas de `run.py` no se veían afectadas.

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
