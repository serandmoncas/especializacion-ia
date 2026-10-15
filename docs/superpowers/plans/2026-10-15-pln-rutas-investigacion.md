# PLN · Motor de rutas de investigación — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el módulo S2 «PLN» (paquete `research_paths`, 5 notebooks smoke + 1 no-smoke,
6 ejercicios) y su mini-proyecto: un motor que, dado un tema, devuelve una ruta de conceptos
(Wikipedia ES, prerrequisitos aprendidos) y lecturas por concepto (OpenAlex), evaluado por capas con
cuatro hipótesis.

**Architecture:** Piezas desde cero en NumPy/Python puro (BPE, atención, BM25, RefD, métricas) con
tests contra la librería de referencia. Clientes HTTP con caché en disco para Wikipedia y OpenAlex
(la red solo se toca una vez). Prerrequisitos: RefD, embeddings + logística, cross-encoder XLM-R y
juez LLM. Lecturas: candidatos por tema y un orden «empieza aquí». `experiments.py` contiene la
lógica de los experimentos (inyectable para el smoke); `proyecto/run.py` es una envoltura delgada.

**Tech Stack:** Python 3.12, NumPy, pandas, scikit-learn, PyTorch (MPS), transformers 5.19,
tokenizers 0.23, sentence-transformers 6.1, anthropic 1.14; uv, ruff, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-pln-rutas-investigacion-design.md`

**Módulo:** `M = 02-semestre-2/02-pln` (todas las rutas relativas a la raíz del repo).

## Global Constraints

- Idioma: español para docs y comentarios; nombres de código en inglés (CLAUDE.md).
- Gestor uv; nunca pip. Antes de cada commit:
  `uv run ruff format && uv run ruff check --fix && uv run pytest` (formatear antes de chequear;
  nunca encadenar `pytest | tail` ni `ruff check | tail` antes de `git commit`).
- Línea máxima 100 (ruff). Nunca `sys.path.insert`. `itertools.pairwise` en vez de `zip(x, x[1:])`.
- Aleatoriedad: semilla 42; generadores explícitos (`np.random.default_rng(seed)`), nunca
  `np.random.*` global; notebooks con `rng = common.set_seed(42)`.
- Datos, caché y pesos nunca al repo (`**/data/*` ya está ignorado); `data/download.py` descarga
  AL-CPL con commit fijo y SHA-256.
- Claves solo desde el entorno: `OPENALEX_API_KEY`, `ANTHROPIC_API_KEY`. Nunca en código, caché,
  logs, mensajes de error ni `metrics.json`.
- CI: `uv sync --locked --extra base --extra dl --extra nlp` (se agrega `nlp` en la Tarea 1); el CI
  no tiene red hacia modelos ni APIs: tests y notebooks smoke no descargan nada.
- Modelos: `xlm-roberta-base`, `intfloat/multilingual-e5-base` (prefijos `query: `/`passage: `),
  `claude-haiku-5-5`. Par de textos truncado a 256 tokens (`longest_first`).
- Hipótesis: bootstrap por conglomerados, 2 000 réplicas, IC 95 %, p unilateral
  = max(fracción de réplicas ≤ 0, 1/2 000), Holm sobre H1–H4; «se apoya» = p Holm < 0,05 **y**
  Δ > 0.
- `metrics.json` se escribe con `allow_nan=False` (en Visión se coló un NaN).
- Trabajo en la rama `pln-rutas` (no en main); preguntar al usuario antes del merge.
- TODO del usuario: `order_route` en `concepts.py` (versión mínima: contar victorias). No completarlo.
- Desvíos deliberados de la spec §3 y §6 (nombres y un ejercicio): `tokenize.py` → `bpe.py` (no
  confundir con el módulo estándar `tokenize`); `prereq/models.py` → `prereq/features.py` +
  `textclf.py` (sirve también al clasificador introductorio) + `prereq/llm.py`; el ejercicio de
  «máscara causal» pasa a **máscara de relleno** porque `causal_mask` ya está en `src/` y en el
  notebook 03. La lógica de experimentos vive en `experiments.py` para que el smoke la ejecute.

## Hechos verificados antes de escribir el plan (2026-10-15)

- **AL-CPL** commit `11397eaa3ea01c5e06fefcfc1c02a84f21164081`. SHA-256:

  | Archivo | SHA-256 |
  |---|---|
  | data_mining.pairs | 9b0d42ac047f7e4b09f77210afcce5b4639f026c608d31f1d49e71c6b718e5da |
  | data_mining.preqs | 8dda7ec59e7199531baad22545d938acb840a20306ed79ae767e82dfb27d62de |
  | geometry.pairs | 883d6fa4622de6c2fc4581b835e3e1d3ab7d1ad7df00728c070897a0d09ec12c |
  | geometry.preqs | e45787dfe417035c72a2a9cfa984ee7b75193d7b1f27e43a6023b363c9c69c40 |
  | physics.pairs | 639070062f36c80e9393367f20f97e22b98ad3a36edbe36a5999ac930ddd13c4 |
  | physics.preqs | e1ce25603a7f3388ca2d7da034f7de7223eb40e62591415525687718fc059643 |
  | precalculus.pairs | 20de87fbb394410c400871904d50cb723e5c88d31e12bdcc5919933478ba97cd |
  | precalculus.preqs | 80f8034f323d7853cd1036108292572115ebfdb52de78beeee493a5ca15b3015 |

  CSV sin encabezado `A,B` con títulos EN (guiones bajos); en (A, B) B es prerrequisito de A; todos
  los `.preqs` están contenidos en `.pairs`. 545 conceptos únicos, 40 en más de un dominio.
- **Mapeo EN→ES** (`langlinks`, `redirects=1`): 1 concepto sin página EN, 39 sin página ES; quedan
  6 067 de 6 529 pares (data_mining 644, geometry 1 627, physics 1 860, precalculus 1 936) y 1 866
  positivos; 9 colisiones (dos conceptos EN → la misma página ES).
- **Wikipedia** responde **429 sin un User-Agent con contacto**; con
  `especializacion-ia/0.1 (https://github.com/serandmoncas/especializacion-ia; ...)` funciona.
  `extracts` con `exintro` admite hasta 20 títulos por petición; «Contraste de hipótesis» tiene 174
  palabras de introducción y 46 enlaces.
- **OpenAlex:** sin clave hay presupuesto diario compartido (0,10 USD); con `api_key` 1 USD/día.
  ~44 % de los resultados trae `abstract_inverted_index`. `filter=primary_topic.id:T…,type:review,
  publication_year:2010-2024,referenced_works_count:>19` funciona; 4 516 temas.
- **BPE propio = `tokenizers.BpeTrainer`** en 200/200 corpus aleatorios si el empate se rompe por el
  menor (id_a, id_b), con ids = alfabeto ordenado y luego orden de creación; `encode` aplicando
  siempre la fusión de menor rango coincide en 100/100. WordPiece greedy (más largo primero, `##`)
  coincide con `tokenizers.models.WordPiece`.
- **XLM-R base en MPS (M4):** 1,2 s por lote de 16 a 256 tokens; carga 45 s la primera vez.
  Un `XLMRobertaForSequenceClassification` diminuto con un `PreTrainedTokenizerFast` entrenado en
  local (sin descargas) corre en 0,02 s: así se prueba el ajuste en CI.
- `transformers` 5.19 imprime un «LOAD REPORT» con `UNEXPECTED`/`MISSING` al cargar XLM-R para
  clasificación: es esperado (cabeza nueva).

## Review Focus

1. Tema de la CLI que cae en una **página de desambiguación** («Mercurio») o en una redirección:
   debe sugerir títulos, no construir la ruta con los enlaces de la desambiguación (Tarea 4 y 10).
2. Trabajos de OpenAlex con **campos nulos** (`abstract_inverted_index`, `cited_by_count`,
   `publication_year`, `title` en `None`): no deben romper nada; se tratan como vacíos/0 (Tarea 4 y 9).
3. **Errores transitorios de Anthropic** (429/529/sobrecarga) a mitad de la evaluación: reintento y
   caché por respuesta, para que volver a correr reanude sin pagar dos veces (Tarea 8).
4. **Fuga de la clave** en el nombre de un archivo de caché, en su contenido o en un mensaje de error
   (Tarea 4).
5. **NaN en `metrics.json`** (un dominio o tema sin positivos, una métrica indefinida): se escribe
   con `allow_nan=False` y las métricas devuelven valores definidos (Tarea 3 y 11).

## Estructura de archivos

```
M/
├── README.md                         # Tarea 14 (reescrito)
├── data/download.py                  # Tarea 5 (AL-CPL)
├── src/research_paths/
│   ├── __init__.py                   # Tarea 1
│   ├── bpe.py                        # Tarea 1
│   ├── attention.py                  # Tarea 2
│   ├── retrieval.py                  # Tarea 3 (BM25, codificadores, coseno)
│   ├── evaluation.py                 # Tarea 3 (F1, AP, umbral, recall@k, NDCG, bootstrap, Holm)
│   ├── http.py                       # Tarea 4 (caché + reintentos)
│   ├── wiki.py                       # Tarea 4
│   ├── openalex.py                   # Tarea 4
│   ├── alcpl.py                      # Tarea 5
│   ├── prereq/__init__.py            # Tarea 6
│   ├── prereq/refd.py                # Tarea 6
│   ├── prereq/features.py            # Tarea 6 (embeddings + logística)
│   ├── textclf.py                    # Tarea 7 (XLM-R afinado: pares o un texto)
│   ├── prereq/llm.py                 # Tarea 8
│   ├── readings.py                   # Tarea 9
│   ├── concepts.py                   # Tarea 10
│   ├── cli.py, __main__.py           # Tarea 10
│   └── experiments.py                # Tarea 11
├── tests/                            # una prueba por archivo de src + smoke
├── notebooks/01…06                   # Tarea 12
├── ejercicios/ej01…ej06 + soluciones # Tarea 13
└── proyecto/run.py, README.md, resultados/   # Tareas 11 y 14
```

---

### Task 1: Andamiaje del paquete y BPE desde cero

**Files:**
- Modify: `pyproject.toml` (lista `packages`)
- Modify: `.github/workflows/ci.yml` (paso «Instalar dependencias»)
- Create: `M/src/research_paths/__init__.py`, `M/src/research_paths/bpe.py`
- Create: `M/tests/test_bpe.py`
- Delete: `M/src/.gitkeep`, `M/tests/.gitkeep`

**Interfaces:**
- Produces: `train_bpe(corpus: Iterable[str], num_merges: int) -> list[tuple[str, str]]`;
  `encode(text: str, merges: list[tuple[str, str]]) -> list[str]`.

- [ ] **Step 0: Rama**

```bash
git checkout -b pln-rutas
```

- [ ] **Step 1: Registrar el paquete y el extra nlp en CI**

En `pyproject.toml`, agregar al final de `packages` (después de la línea de `applied_cv`):

```toml
    "02-semestre-2/02-pln/src/research_paths",
```

En `.github/workflows/ci.yml` (verificar el nombre del archivo con `ls .github/workflows`), cambiar:

```yaml
      - name: Instalar dependencias (extras base y dl, grupo dev)
        run: uv sync --locked --extra base --extra dl  # torch solo-CPU en Linux (ver pyproject)
```

por:

```yaml
      - name: Instalar dependencias (extras base, dl y nlp, grupo dev)
        run: uv sync --locked --extra base --extra dl --extra nlp  # torch solo-CPU en Linux
```

Crear `M/src/research_paths/__init__.py`:

```python
"""Motor de rutas de investigación: conceptos en orden (Wikipedia) y lecturas (OpenAlex)."""
```

```bash
git rm -q 02-semestre-2/02-pln/src/.gitkeep 02-semestre-2/02-pln/tests/.gitkeep
uv sync --extra base --extra dl --extra nlp --reinstall-package especializacion-ia
```

- [ ] **Step 2: Escribir los tests de BPE**

`M/tests/test_bpe.py`:

```python
"""BPE propio contra `tokenizers` de Hugging Face (mismas fusiones y misma tokenización)."""

import json
import random

import pytest

from research_paths.bpe import encode, train_bpe

WORDS = [
    "casa", "casas", "cosa", "cosas", "perro", "perros", "gato", "gatos", "camino", "caminos",
    "canto", "cantos", "tema", "temas", "investigación", "información", "aprendizaje",
]  # fmt: skip


def test_first_merge_is_the_most_frequent_pair():
    assert train_bpe(["ab ab ab aa"], 1) == [("a", "b")]


def test_ties_break_by_smallest_symbol_ids():
    # «aa» y «ab» aparecen 3 veces; gana (a, a) porque id(a) = 0 < id(b) = 1.
    assert train_bpe(["aaab aab ab"], 1) == [("a", "a")]


def test_stops_when_no_pairs_remain():
    assert train_bpe(["ab"], 10) == [("a", "b")]


def test_encode_reconstructs_each_word_and_compresses():
    merges = train_bpe([" ".join(WORDS * 3)], 30)
    text = "gatos caminos información"
    tokens = encode(text, merges)
    assert "".join(tokens) == text.replace(" ", "")
    assert len(tokens) < len(text.replace(" ", ""))


def _hf_bpe(corpus: list[str], num_merges: int):
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer(tokenizers.models.BPE())
    tok.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    alphabet = {ch for line in corpus for ch in line if not ch.isspace()}
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=len(alphabet) + num_merges, min_frequency=0, show_progress=False
    )
    tok.train_from_iterator(corpus, trainer)
    merges = [tuple(m) for m in json.loads(tok.to_str())["model"]["merges"]]
    return tok, merges


def test_merges_and_tokens_match_huggingface_tokenizers():
    rng = random.Random(0)
    for _ in range(50):
        corpus = [" ".join(rng.choices(WORDS, k=rng.randint(20, 300)))]
        tok, ref = _hf_bpe(corpus, 25)
        merges = train_bpe(corpus, 25)
        assert merges == ref
        text = " ".join(
            w for w in "gatos cantan caminos de información".split() if set(w) <= set(corpus[0])
        )
        assert encode(text, merges) == tok.encode(text).tokens
```

- [ ] **Step 3: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_bpe.py -q`
Expected: FAIL / ERROR con `ModuleNotFoundError: No module named 'research_paths.bpe'`.

- [ ] **Step 4: Implementar `bpe.py`**

`M/src/research_paths/bpe.py`:

```python
"""BPE (byte-pair encoding) desde cero, con el mismo desempate que `tokenizers` de Hugging Face."""

from collections import Counter
from collections.abc import Iterable
from itertools import pairwise

Merge = tuple[str, str]


def _apply(symbols: tuple[str, ...], pair: Merge) -> tuple[str, ...]:
    """Fusiona, de izquierda a derecha, cada aparición de `pair`."""
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(corpus: Iterable[str], num_merges: int) -> list[Merge]:
    """Aprende hasta `num_merges` fusiones; las palabras se separan por espacios.

    En cada paso fusiona el par adyacente más frecuente. En empate gana el de ids menores: el id de
    un símbolo es su posición en el alfabeto ordenado y, para los fusionados, su orden de creación
    (igual que `tokenizers.trainers.BpeTrainer`).
    """
    words = Counter(w for line in corpus for w in line.split())
    vocab: dict[tuple[str, ...], int] = {tuple(w): c for w, c in words.items()}
    ids = {ch: i for i, ch in enumerate(sorted({ch for w in words for ch in w}))}
    merges: list[Merge] = []
    for _ in range(num_merges):
        pairs: Counter[Merge] = Counter()
        for symbols, count in vocab.items():
            for pair in pairwise(symbols):
                pairs[pair] += count
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], -ids[kv[0][0]], -ids[kv[0][1]]))[0]
        merges.append(best)
        ids.setdefault(best[0] + best[1], len(ids))
        merged: dict[tuple[str, ...], int] = {}
        for symbols, count in vocab.items():
            new = _apply(symbols, best)
            merged[new] = merged.get(new, 0) + count
        vocab = merged
    return merges


def encode(text: str, merges: list[Merge]) -> list[str]:
    """Tokeniza cada palabra aplicando siempre la fusión de menor rango que esté presente."""
    rank = {pair: i for i, pair in enumerate(merges)}
    tokens: list[str] = []
    for word in text.split():
        symbols = tuple(word)
        while len(symbols) > 1:
            present = [rank[p] for p in pairwise(symbols) if p in rank]
            if not present:
                break
            symbols = _apply(symbols, merges[min(present)])
        tokens.extend(symbols)
    return tokens
```

- [ ] **Step 5: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_bpe.py -q`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add pyproject.toml uv.lock .github/workflows 02-semestre-2/02-pln
git commit -m "feat(pln): paquete research_paths y BPE desde cero (igual a tokenizers)"
```

---

### Task 2: Atención escalada y multi-cabeza en NumPy

**Files:**
- Create: `M/src/research_paths/attention.py`
- Create: `M/tests/test_attention.py`

**Interfaces:**
- Produces: `softmax(x, axis=-1)`; `scaled_dot_product_attention(q, k, v, mask=None) ->
  (out, weights)` con `mask` booleano (True = se puede atender); `causal_mask(n) -> bool (n, n)`;
  `multi_head_attention(x, w_q, w_k, w_v, w_o, num_heads, mask=None) -> (out, weights)` con pesos
  de forma `(..., heads, n, n)` y convención `x @ w`.

- [ ] **Step 1: Tests**

`M/tests/test_attention.py`:

```python
"""Atención propia contra `torch.nn.functional.scaled_dot_product_attention` y `MultiheadAttention`."""

import numpy as np
import pytest

from research_paths.attention import (
    causal_mask,
    multi_head_attention,
    scaled_dot_product_attention,
    softmax,
)


def test_softmax_rows_sum_to_one_and_are_stable():
    p = softmax(np.array([[1000.0, 1000.0], [0.0, -np.inf]]))
    np.testing.assert_allclose(p, [[0.5, 0.5], [1.0, 0.0]])


def test_causal_mask_is_lower_triangular():
    np.testing.assert_array_equal(causal_mask(3), [[1, 0, 0], [1, 1, 0], [1, 1, 1]])


def test_causal_attention_ignores_the_future():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(4, 3)) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(4))
    assert np.all(w[np.triu_indices(4, 1)] == 0)
    np.testing.assert_allclose(w.sum(-1), 1)


def test_matches_torch_sdpa():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(2, 5, 8)) for _ in range(3))
    for mask, causal in ((None, False), (causal_mask(5), True)):
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        ref = torch.nn.functional.scaled_dot_product_attention(
            *(torch.tensor(t) for t in (q, k, v)), is_causal=causal
        )
        np.testing.assert_allclose(out, ref.numpy(), atol=1e-10)


def test_multi_head_matches_torch_multihead_attention():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    d, heads = 8, 2
    x = rng.normal(size=(3, 5, d))
    w_q, w_k, w_v, w_o = (rng.normal(size=(d, d)) for _ in range(4))
    mha = torch.nn.MultiheadAttention(d, heads, bias=False, batch_first=True, dtype=torch.float64)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.tensor(np.concatenate([w_q.T, w_k.T, w_v.T])))
        mha.out_proj.weight.copy_(torch.tensor(w_o.T))
    xt = torch.tensor(x)
    ref, ref_w = mha(xt, xt, xt, average_attn_weights=False)
    out, w = multi_head_attention(x, w_q, w_k, w_v, w_o, heads)
    np.testing.assert_allclose(out, ref.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(w, ref_w.detach().numpy(), atol=1e-10)


def test_multi_head_rejects_indivisible_heads():
    x = np.zeros((2, 6))
    w = np.eye(6)
    with pytest.raises(ValueError, match="divisible"):
        multi_head_attention(x, w, w, w, w, num_heads=4)
```

- [ ] **Step 2: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_attention.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'research_paths.attention'`.

- [ ] **Step 3: Implementar**

`M/src/research_paths/attention.py`:

```python
"""Atención escalada y multi-cabeza en NumPy (la referencia en los tests es `torch`)."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax estable: resta el máximo antes de exponenciar."""
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """softmax(q·kᵀ/√d)·v. `mask` booleano: True = se puede atender. Devuelve (salida, pesos)."""
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


def causal_mask(n: int) -> np.ndarray:
    """Cada posición atiende solo a sí misma y a las anteriores."""
    return np.tril(np.ones((n, n), dtype=bool))


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Auto-atención con `num_heads` cabezas; `x` de forma (..., n, d) y proyecciones `x @ w`."""
    *lead, n, d = x.shape
    if d % num_heads:
        raise ValueError(f"d = {d} no es divisible entre {num_heads} cabezas")
    h = d // num_heads

    def split(t: np.ndarray) -> np.ndarray:  # (..., n, d) -> (..., cabezas, n, h)
        return np.swapaxes(t.reshape(*lead, n, num_heads, h), -2, -3)

    out, weights = scaled_dot_product_attention(split(x @ w_q), split(x @ w_k), split(x @ w_v), mask)
    out = np.swapaxes(out, -2, -3).reshape(*lead, n, d)
    return out @ w_o, weights
```

- [ ] **Step 4: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_attention.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): atención escalada y multi-cabeza en NumPy (verificada contra torch)"
```

---

### Task 3: Recuperación (BM25, codificadores) y métricas de evaluación

**Files:**
- Create: `M/src/research_paths/retrieval.py`, `M/src/research_paths/evaluation.py`
- Create: `M/tests/test_retrieval.py`, `M/tests/test_evaluation.py`

**Interfaces:**
- Produces (retrieval): `simple_tokenize(text) -> list[str]`; `BM25(k1=1.5, b=0.75).fit(docs:
  list[list[str]]) -> BM25`, `.scores(query: list[str]) -> np.ndarray`; `Encoder` =
  `Callable[[list[str], str], np.ndarray]` (segundo argumento `kind` ∈ {"query", "passage"}, filas
  normalizadas L2); `hashing_encoder(dim=256) -> Encoder`; `e5_encoder(model_name=..., device=None,
  batch_size=32) -> Encoder`; `cosine_scores(query_vec, doc_mat) -> np.ndarray`.
- Produces (evaluation): `f1_binary(y_true, y_pred) -> float`; `average_precision(y_true, scores)
  -> float`; `best_threshold(y_true, scores) -> float`; `recall_at_k(ranked, relevant, k) -> float`;
  `ndcg_at_k(ranked, relevant, k) -> float`; `cluster_bootstrap(clusters, stat, rng, n_boot=2000)
  -> {"delta", "ci", "p"}`; `holm(pvalues) -> list[float]`.

- [ ] **Step 1: Tests de recuperación**

`M/tests/test_retrieval.py`:

```python
"""BM25 contra un cálculo a mano; codificador de hashing y coseno."""

import math

import numpy as np

from research_paths.retrieval import BM25, cosine_scores, hashing_encoder, simple_tokenize


def test_simple_tokenize_keeps_accents_and_lowercases():
    assert simple_tokenize("Información, ¡Aprendizaje!") == ["información", "aprendizaje"]


def test_bm25_matches_hand_computation():
    docs = [["a", "b"], ["a"], ["c", "c", "a"]]
    bm = BM25().fit(docs)
    # N = 3, n(c) = 1, avgdl = 2, |d3| = 3, f = 2
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2))
    np.testing.assert_allclose(bm.scores(["c"]), [0, 0, expected])


def test_bm25_ranks_the_matching_document_first_and_ignores_unknown_terms():
    docs = [simple_tokenize(t) for t in ("el gato negro", "el perro", "aprendizaje por refuerzo")]
    s = BM25().fit(docs).scores(simple_tokenize("refuerzo xyz"))
    assert s.argmax() == 2 and s[0] == s[1] == 0


def test_bm25_common_term_has_small_positive_idf():
    bm = BM25().fit([["a"], ["a", "b"]])
    assert 0 < bm.idf["a"] < bm.idf["b"]


def test_hashing_encoder_is_normalized_and_deterministic():
    enc = hashing_encoder(64)
    a = enc(["aprendizaje por refuerzo", ""], "query")
    np.testing.assert_allclose(np.linalg.norm(a[0]), 1)
    assert np.all(a[1] == 0)
    np.testing.assert_array_equal(a, enc(["aprendizaje por refuerzo", ""], "passage"))


def test_cosine_scores_rank_similar_text_first():
    enc = hashing_encoder(256)
    docs = enc(["gato negro", "aprendizaje por refuerzo profundo", "perro"], "passage")
    q = enc(["refuerzo profundo"], "query")[0]
    assert cosine_scores(q, docs).argmax() == 1


def test_cosine_scores_handles_zero_vectors():
    assert np.all(cosine_scores(np.zeros(3), np.zeros((2, 3))) == 0)

```

- [ ] **Step 2: Tests de evaluación**

`M/tests/test_evaluation.py`:

```python
"""Métricas propias contra scikit-learn y statsmodels; bootstrap por conglomerados."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, ndcg_score
from statsmodels.stats.multitest import multipletests

from research_paths.evaluation import (
    average_precision,
    best_threshold,
    cluster_bootstrap,
    f1_binary,
    holm,
    ndcg_at_k,
    recall_at_k,
)


def test_f1_matches_sklearn_and_is_zero_without_true_positives():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y, p = rng.integers(0, 2, 40), rng.integers(0, 2, 40)
        assert f1_binary(y, p) == pytest.approx(f1_score(y, p, zero_division=0))
    assert f1_binary(np.zeros(5), np.zeros(5)) == 0.0


def test_average_precision_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(100):
        y = rng.integers(0, 2, 30)
        y[0] = 1
        s = np.round(rng.random(30), 1)  # muchos empates
        assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))


def test_average_precision_without_positives_is_zero_not_nan():
    assert average_precision(np.zeros(4), np.arange(4.0)) == 0.0


def test_best_threshold_maximizes_f1():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.7, 0.75, 0.9])
    t = best_threshold(y, s)
    best = max(f1_binary(y, s >= c) for c in np.unique(s))
    assert f1_binary(y, s >= t) == best


def test_best_threshold_without_positives_predicts_nothing():
    assert best_threshold(np.zeros(3), np.array([0.1, 0.5, 0.9])) == np.inf


def test_recall_and_ndcg_at_k():
    ranked = ["a", "x", "b", "y"]
    assert recall_at_k(ranked, {"a", "b", "z"}, 2) == pytest.approx(1 / 3)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(ranked, set(), 2)


def test_ndcg_matches_sklearn_when_all_relevant_are_ranked():
    rng = np.random.default_rng(2)
    for _ in range(50):
        ids = [f"w{i}" for i in range(15)]
        rel = {i for i in ids if rng.random() < 0.3} or {"w3"}
        scores = np.sort(rng.random(15))[::-1]
        y_true = np.array([[1.0 if i in rel else 0.0 for i in ids]])
        ref = ndcg_score(y_true, scores[None, :], k=10)
        if len(rel) <= 10:
            assert ndcg_at_k(ids, rel, 10) == pytest.approx(ref)


def test_holm_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.2]
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1])


def test_cluster_bootstrap_resamples_whole_clusters():
    clusters = np.array([0, 1, 1, 1])
    out = cluster_bootstrap(clusters, lambda idx: float(len(idx)), np.random.default_rng(0), 200)
    assert out["delta"] == 4.0
    assert set(np.unique(out["_boots"])) <= {2.0, 4.0, 6.0}


def test_cluster_bootstrap_p_value_floor_and_direction():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(30), 3)
    diff = rng.normal(1.0, 0.1, 90)
    out = cluster_bootstrap(clusters, lambda idx: float(diff[idx].mean()), rng, 2000)
    assert out["p"] == pytest.approx(1 / 2000)
    assert out["ci"][0] < out["delta"] < out["ci"][1]
```

- [ ] **Step 3: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_retrieval.py 02-semestre-2/02-pln/tests/test_evaluation.py -q`
Expected: ERROR por `ModuleNotFoundError` de `research_paths.retrieval` y `research_paths.evaluation`.

- [ ] **Step 4: Implementar `retrieval.py`**

```python
"""Recuperación: BM25 desde cero, codificadores de texto y orden denso por coseno."""

import math
import re
import zlib
from collections import Counter
from collections.abc import Callable

import numpy as np

_WORD = re.compile(r"\w+", re.UNICODE)

# encode(textos, kind) -> matriz (n, d) con filas de norma 1 (o 0 si el texto está vacío);
# kind ∈ {"query", "passage"} (e5 usa prefijos distintos).
Encoder = Callable[[list[str], str], np.ndarray]


def simple_tokenize(text: str) -> list[str]:
    """Minúsculas y palabras Unicode (conserva tildes)."""
    return _WORD.findall(text.lower())


class BM25:
    """Okapi BM25 con idf = ln(1 + (N − n + 0,5)/(n + 0,5)), siempre positivo (como Lucene)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.tf = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        mean = float(self.lengths.mean()) if len(docs) else 0.0
        self.avgdl = mean if mean > 0 else 1.0
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        return self

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in query:
            idf = self.idf.get(term)
            if idf is None:
                continue
            f = np.array([tf[term] for tf in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def hashing_encoder(dim: int = 256) -> Encoder:
    """Bolsa de palabras con hashing (sin modelo): para tests y notebooks; ignora `kind`."""

    def encode(texts: list[str], kind: str = "query") -> np.ndarray:
        out = np.zeros((len(texts), dim))
        for i, text in enumerate(texts):
            for word in simple_tokenize(text):
                out[i, zlib.crc32(word.encode()) % dim] += 1.0
        return _normalize_rows(out)

    return encode


def e5_encoder(
    model_name: str = "intfloat/multilingual-e5-base", device: str | None = None, batch_size: int = 32
) -> Encoder:
    """e5 multilingüe vía sentence-transformers (se descarga la primera vez, ~1,1 GB)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str], kind: str = "query") -> np.ndarray:
        prefix = "query: " if kind == "query" else "passage: "
        return model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return encode


def cosine_scores(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    """Coseno entre un vector y cada fila; los vectores nulos dan 0."""
    q = np.asarray(query_vec, dtype=float)
    q = q / (np.linalg.norm(q) or 1.0)
    return _normalize_rows(np.asarray(doc_mat, dtype=float)) @ q
```

- [ ] **Step 5: Implementar `evaluation.py`**

```python
"""Métricas (F1, AP, recall@k, NDCG@k), umbral por F1, bootstrap por conglomerados y Holm."""

from collections.abc import Callable

import numpy as np


def f1_binary(y_true, y_pred) -> float:
    """F1 de la clase positiva; 0 si no hay verdaderos positivos."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    if tp == 0:
        return 0.0
    precision = tp / int(np.sum(y_pred))
    recall = tp / int(np.sum(y_true))
    return 2 * precision * recall / (precision + recall)


def average_precision(y_true, scores) -> float:
    """AP como `sklearn.metrics.average_precision_score` (empates agrupados); 0 sin positivos."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    y, s = y_true[order], scores[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # último índice de cada umbral
    tps = np.cumsum(y)[last]
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_threshold(y_true, scores) -> float:
    """Umbral (predecir `scores >= t`) con F1 máximo; en empate, el mayor. Sin positivos: +inf."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    if not y_true.any():
        return float("inf")
    best_t, best_f1 = float("inf"), -1.0
    for t in np.unique(scores)[::-1]:
        f1 = f1_binary(y_true, scores >= t)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("el conjunto relevante está vacío")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """NDCG binario; el ideal pone todos los relevantes (aunque no estén en `ranked`) arriba."""
    if not relevant:
        raise ValueError("el conjunto relevante está vacío")
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(dcg / ideal)


def cluster_bootstrap(
    clusters, stat: Callable[[np.ndarray], float], rng: np.random.Generator, n_boot: int = 2000
) -> dict:
    """Remuestrea conglomerados completos y recalcula `stat(índices)`.

    Devuelve la diferencia observada, el IC 95 % percentil y p unilateral (H1: Δ > 0)
    = max(fracción de réplicas ≤ 0, 1/n_boot). `_boots` queda para diagnóstico.
    """
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    observed = float(stat(np.arange(len(clusters))))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots[i] = stat(np.concatenate([members[c] for c in pick]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = max(float(np.mean(boots <= 0)), 1 / n_boot)
    return {"delta": observed, "ci": [float(lo), float(hi)], "p": p, "_boots": boots}


def holm(pvalues) -> list[float]:
    """Corrección de Holm-Bonferroni (escalonada, monótona y acotada en 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()
```

- [ ] **Step 6: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_retrieval.py 02-semestre-2/02-pln/tests/test_evaluation.py -q`
Expected: `17 passed`.

- [ ] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): BM25, codificadores y métricas (AP, NDCG, bootstrap por conglomerados, Holm)"
```

---

### Task 4: Clientes HTTP con caché: Wikipedia y OpenAlex

**Files:**
- Create: `M/src/research_paths/http.py`, `M/src/research_paths/wiki.py`,
  `M/src/research_paths/openalex.py`
- Create: `M/tests/conftest.py` (servidor falso como fixture `server_cls`), `M/tests/test_http.py`,
  `M/tests/test_wiki.py`, `M/tests/test_openalex.py`

**Interfaces:**
- Produces (http): `USER_AGENT`; `HTTPError(status, url)`; `DiskCache(root).get(key) / .set(key,
  value)`; `CachedClient(cache_dir, fetch=urllib_fetch, max_retries=5, base_delay=1.0,
  sleep=time.sleep)` con `.get_json(base_url, params) -> dict` y `.stats = {"network", "cache"}`;
  `MissingKeyError`; `require_env(name, where) -> str`.
- Produces (wiki): `Wiki(client, lang="es")` con `.resolve(titles) -> dict[str, str | None]`
  (None = no existe o es desambiguación), `.extracts(titles) -> dict[str, str]`,
  `.links(title) -> list[str]`, `.langlinks(titles, target) -> dict[str, str | None]`,
  `.search(query, limit=5) -> list[str]`.
- Produces (openalex): `WORK_FIELDS`; `OpenAlex(client, api_key)` con `.works(*, filter, search=None,
  sort=None, max_results=200, select=WORK_FIELDS) -> list[dict]` y `.topics(*, filter) -> list[dict]`;
  `abstract_text(inv) -> str`; `short_id(x) -> str`.

- [ ] **Step 1: Servidor falso para los tests**

`M/tests/conftest.py`:

```python
"""Servidor HTTP falso para los tests (respuestas grabadas a mano), expuesto como fixture."""

import json
import urllib.parse

import pytest


class FakeServer:
    """`routes` es una lista de (predicado(params) -> bool, respuesta | [respuestas]).

    Una respuesta es un dict (200 con ese JSON) o un int (código de error). Si hay una lista, se
    consume en orden (sirve para simular 429 y luego 200).
    """

    def __init__(self, routes):
        self.routes = [(pred, resp if isinstance(resp, list) else [resp]) for pred, resp in routes]
        self.calls: list[str] = []

    def __call__(self, url: str) -> tuple[int, bytes]:
        self.calls.append(url)
        params = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).items()}
        for pred, responses in self.routes:
            if pred(params):
                resp = responses.pop(0) if len(responses) > 1 else responses[0]
                if isinstance(resp, int):
                    return resp, b""
                return 200, json.dumps(resp).encode()
        return 404, b""


@pytest.fixture
def server_cls():
    return FakeServer
```

- [ ] **Step 2: Tests de `http.py`**

`M/tests/test_http.py`:

```python
"""Caché en disco, reintentos y que la clave nunca se filtre."""

import pytest
from research_paths.http import CachedClient, HTTPError, MissingKeyError, require_env

SECRET = "sk-secreto-123"


def _client(tmp_path, server, **kw):
    sleeps = []
    client = CachedClient(tmp_path / "cache", fetch=server, sleep=sleeps.append, **kw)
    return client, sleeps


def test_second_call_comes_from_cache(tmp_path, server_cls):
    server = server_cls([(lambda p: True, {"ok": 1})])
    client, _ = _client(tmp_path, server)
    assert client.get_json("https://x.org/api", {"q": "a"}) == {"ok": 1}
    assert client.get_json("https://x.org/api", {"q": "a"}) == {"ok": 1}
    assert len(server.calls) == 1
    assert client.stats == {"network": 1, "cache": 1}


def test_retries_429_with_exponential_backoff(tmp_path, server_cls):
    server = server_cls([(lambda p: True, [429, 503, {"ok": 2}])])
    client, sleeps = _client(tmp_path, server)
    assert client.get_json("https://x.org/api", {}) == {"ok": 2}
    assert sleeps == [1.0, 2.0]


def test_client_error_fails_immediately_and_is_not_cached(tmp_path, server_cls):
    server = server_cls([(lambda p: True, [404, {"ok": 3}])])
    client, _ = _client(tmp_path, server)
    with pytest.raises(HTTPError) as err:
        client.get_json("https://x.org/api", {"api_key": SECRET})
    assert err.value.status == 404
    assert SECRET not in str(err.value)
    assert client.get_json("https://x.org/api", {"api_key": SECRET}) == {"ok": 3}


def test_gives_up_after_max_retries(tmp_path, server_cls):
    server = server_cls([(lambda p: True, 503)])
    client, sleeps = _client(tmp_path, server, max_retries=3)
    with pytest.raises(HTTPError, match="503"):
        client.get_json("https://x.org/api", {})
    assert len(server.calls) == 3 and len(sleeps) == 2


def test_api_key_is_sent_but_never_stored(tmp_path, server_cls):
    server = server_cls([(lambda p: p.get("api_key") == SECRET, {"ok": 4})])
    client, _ = _client(tmp_path, server)
    assert client.get_json("https://x.org/api", {"api_key": SECRET, "q": "b"}) == {"ok": 4}
    files = list((tmp_path / "cache").rglob("*"))
    assert files
    for f in files:
        assert SECRET not in f.name
        if f.is_file():
            assert SECRET not in f.read_text()


def test_require_env_explains_where_to_get_the_key(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    with pytest.raises(MissingKeyError, match=r"OPENALEX_API_KEY.*openalex.org"):
        require_env("OPENALEX_API_KEY", "https://openalex.org/settings/api")
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    assert require_env("OPENALEX_API_KEY", "x") == "k"
```

- [ ] **Step 3: Tests de `wiki.py`**

`M/tests/test_wiki.py`:

```python
"""Cliente de Wikipedia con respuestas grabadas: redirecciones, desambiguación, continuación."""

from research_paths.http import CachedClient
from research_paths.wiki import Wiki


def _wiki(server_cls, tmp_path, routes, lang="es"):
    return Wiki(CachedClient(tmp_path, fetch=server_cls(routes), sleep=lambda s: None), lang)


def test_resolve_follows_normalization_and_redirects_and_rejects_disambiguation(tmp_path, server_cls):
    resp = {
        "query": {
            "normalized": [{"from": "aprendizaje_por_refuerzo", "to": "Aprendizaje por refuerzo"}],
            "redirects": [{"from": "Aprendizaje por refuerzo", "to": "Aprendizaje reforzado"}],
            "pages": [
                {"title": "Aprendizaje reforzado", "pageid": 1},
                {"title": "Mercurio", "pageid": 2, "pageprops": {"disambiguation": ""}},
                {"title": "Nada", "missing": True},
            ],
        }
    }
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("prop") == "pageprops", resp)])
    out = wiki.resolve(["aprendizaje_por_refuerzo", "Mercurio", "Nada"])
    assert out == {"aprendizaje_por_refuerzo": "Aprendizaje reforzado", "Mercurio": None, "Nada": None}


def test_extracts_maps_back_to_input_titles_and_missing_is_empty(tmp_path, server_cls):
    resp = {
        "query": {
            "normalized": [{"from": "Gato_doméstico", "to": "Gato doméstico"}],
            "pages": [
                {"title": "Gato doméstico", "extract": "El gato es un mamífero."},
                {"title": "Nada", "missing": True},
            ],
        }
    }
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("prop") == "extracts", resp)])
    assert wiki.extracts(["Gato_doméstico", "Nada"]) == {
        "Gato_doméstico": "El gato es un mamífero.",
        "Nada": "",
    }


def test_links_follow_continuation(tmp_path, server_cls):
    first = {
        "continue": {"plcontinue": "1|0|B", "continue": "||"},
        "query": {"pages": [{"title": "T", "links": [{"ns": 0, "title": "A"}]}]},
    }
    second = {"query": {"pages": [{"title": "T", "links": [{"ns": 0, "title": "B"}]}]}}
    wiki = _wiki(server_cls, tmp_path,
        [
            (lambda p: p.get("prop") == "links" and "plcontinue" not in p, first),
            (lambda p: p.get("plcontinue") == "1|0|B", second),
        ],
    )
    assert wiki.links("T") == ["A", "B"]


def test_langlinks_returns_none_without_translation(tmp_path, server_cls):
    resp = {
        "query": {
            "pages": [
                {"title": "Null hypothesis", "langlinks": [{"lang": "es", "title": "Hipótesis nula"}]},
                {"title": "AdaBoost"},
            ],
            "normalized": [{"from": "Null_hypothesis", "to": "Null hypothesis"}],
        }
    }
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("prop") == "langlinks", resp)], lang="en")
    assert wiki.langlinks(["Null_hypothesis", "AdaBoost"], "es") == {
        "Null_hypothesis": "Hipótesis nula",
        "AdaBoost": None,
    }


def test_search_returns_titles(tmp_path, server_cls):
    resp = {"query": {"search": [{"title": "Mercurio (planeta)"}, {"title": "Mercurio (elemento)"}]}}
    wiki = _wiki(server_cls, tmp_path, [(lambda p: p.get("list") == "search", resp)])
    assert wiki.search("mercurio", 2) == ["Mercurio (planeta)", "Mercurio (elemento)"]


def test_batches_are_split(tmp_path, server_cls):
    server = server_cls([(lambda p: True, {"query": {"pages": []}})])
    wiki = Wiki(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), "es")
    wiki.extracts([f"T{i}" for i in range(45)])  # 20 + 20 + 5
    assert len(server.calls) == 3
```

- [ ] **Step 4: Tests de `openalex.py`**

`M/tests/test_openalex.py`:

```python
"""Cliente de OpenAlex: paginación por cursor, resumen invertido, ids y campos nulos."""

from research_paths.http import CachedClient
from research_paths.openalex import OpenAlex, abstract_text, short_id


def test_abstract_text_rebuilds_order_and_tolerates_none():
    inv = {"aprendizaje": [1], "El": [0], "refuerzo": [3], "por": [2]}
    assert abstract_text(inv) == "El aprendizaje por refuerzo"
    assert abstract_text(None) == ""
    assert abstract_text({}) == ""


def test_short_id():
    assert short_id("https://openalex.org/W123") == "W123"
    assert short_id("W9") == "W9"


def test_works_paginates_with_cursor_and_stops_at_max(tmp_path, server_cls):
    page1 = {"meta": {"next_cursor": "c2"}, "results": [{"id": f"W{i}"} for i in range(3)]}
    page2 = {"meta": {"next_cursor": "c3"}, "results": [{"id": f"W{i}"} for i in range(3, 6)]}
    server = server_cls(
        [(lambda p: p.get("cursor") == "*", page1), (lambda p: p.get("cursor") == "c2", page2)]
    )
    oa = OpenAlex(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), api_key="k")
    out = oa.works(filter="type:review", max_results=5, per_page=3)
    assert [w["id"] for w in out] == ["W0", "W1", "W2", "W3", "W4"]
    assert all("api_key=k" in c for c in server.calls)


def test_works_stops_when_cursor_is_exhausted(tmp_path, server_cls):
    page = {"meta": {"next_cursor": None}, "results": [{"id": "W1", "title": None}]}
    server = server_cls([(lambda p: True, page)])
    oa = OpenAlex(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), api_key=None)
    assert oa.works(filter="x", max_results=200) == [{"id": "W1", "title": None}]
    assert "api_key" not in server.calls[0]


def test_topics_reads_all_pages(tmp_path, server_cls):
    p1 = {"meta": {"next_cursor": "n"}, "results": [{"id": "T1"}] * 200}
    p2 = {"meta": {"next_cursor": None}, "results": [{"id": "T2"}]}
    server = server_cls([(lambda p: p.get("cursor") == "*", p1), (lambda p: p.get("cursor") == "n", p2)])
    oa = OpenAlex(CachedClient(tmp_path, fetch=server, sleep=lambda s: None), api_key=None)
    assert len(oa.topics(filter="field.id:17")) == 201
```

- [ ] **Step 5: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_http.py 02-semestre-2/02-pln/tests/test_wiki.py 02-semestre-2/02-pln/tests/test_openalex.py -q`
Expected: ERROR `ModuleNotFoundError` (`research_paths.http`, `.wiki`, `.openalex`).

- [ ] **Step 6: Implementar `http.py`**

```python
"""GET con caché en disco y reintentos. Las claves de API nunca entran a la caché ni a los errores."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

# Wikimedia responde 429 a clientes sin un User-Agent con forma de contacto.
USER_AGENT = (
    "especializacion-ia/0.1 (https://github.com/serandmoncas/especializacion-ia; proyecto de estudio)"
)
SECRET_PARAMS = frozenset({"api_key"})
RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 599})

Fetch = Callable[[str], tuple[int, bytes]]


class HTTPError(RuntimeError):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} en {url}")
        self.status = status


class MissingKeyError(RuntimeError):
    pass


def require_env(name: str, where: str) -> str:
    """Lee una clave del entorno o explica cómo conseguirla."""
    value = os.environ.get(name)
    if not value:
        raise MissingKeyError(
            f"Falta {name} en el entorno. Consíguela en {where} y expórtala: export {name}=..."
        )
    return value


class DiskCache:
    """JSON por clave en `root/<sha1[:2]>/<sha1>.json`; escritura atómica."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str):
        path = self._path(key)
        return json.loads(path.read_text()) if path.is_file() else None

    def set(self, key: str, value) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(path)


def _url(base_url: str, params: dict, *, public: bool) -> str:
    items = sorted((k, v) for k, v in params.items() if not (public and k in SECRET_PARAMS))
    return f"{base_url}?{urllib.parse.urlencode(items)}"


def urllib_fetch(url: str) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, b""
    except (urllib.error.URLError, TimeoutError):
        return 599, b""  # sin conexión: se reintenta como un 5xx


class CachedClient:
    def __init__(
        self,
        cache_dir: Path,
        fetch: Fetch = urllib_fetch,
        max_retries: int = 5,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache = DiskCache(cache_dir)
        self.fetch, self.max_retries, self.base_delay, self.sleep = (
            fetch,
            max_retries,
            base_delay,
            sleep,
        )
        self.stats = {"network": 0, "cache": 0}

    def get_json(self, base_url: str, params: dict) -> dict:
        key = _url(base_url, params, public=True)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats["cache"] += 1
            return hit
        full = _url(base_url, params, public=False)
        for attempt in range(self.max_retries):
            status, body = self.fetch(full)
            self.stats["network"] += 1
            if status == 200:
                data = json.loads(body)
                self.cache.set(key, data)
                return data
            if status not in RETRY_STATUS or attempt == self.max_retries - 1:
                raise HTTPError(status, key)
            self.sleep(self.base_delay * 2**attempt)
        raise AssertionError("inalcanzable")
```

- [ ] **Step 7: Implementar `wiki.py`**

```python
"""Cliente mínimo de la API de MediaWiki (Wikipedia) sobre `CachedClient`."""

from itertools import batched

from research_paths.http import CachedClient


class Wiki:
    def __init__(self, client: CachedClient, lang: str = "es"):
        self.client, self.lang = client, lang
        self.url = f"https://{lang}.wikipedia.org/w/api.php"

    def _query_all(self, **params) -> list[dict]:
        """Hace la consulta y sigue `continue` hasta agotarla."""
        base = {"action": "query", "format": "json", "formatversion": 2, **params}
        out, cont = [], {}
        while True:
            data = self.client.get_json(self.url, {**base, **cont})
            out.append(data)
            if "continue" not in data:
                return out
            cont = data["continue"]

    @staticmethod
    def _final_titles(responses: list[dict], titles: list[str]) -> dict[str, str]:
        norm, redir = {}, {}
        for data in responses:
            q = data.get("query", {})
            norm.update({x["from"]: x["to"] for x in q.get("normalized", [])})
            redir.update({x["from"]: x["to"] for x in q.get("redirects", [])})
        out = {}
        for title in titles:
            t, seen = norm.get(title, title), set()
            while t in redir and t not in seen:
                seen.add(t)
                t = redir[t]
            out[title] = t
        return out

    @staticmethod
    def _pages(responses: list[dict]) -> dict[str, dict]:
        """Une las páginas de varias respuestas; las propiedades de lista se concatenan."""
        pages: dict[str, dict] = {}
        for data in responses:
            for page in data.get("query", {}).get("pages", []):
                merged = pages.setdefault(page["title"], {})
                for key, value in page.items():
                    if isinstance(value, list):
                        merged.setdefault(key, []).extend(value)
                    else:
                        merged[key] = value
        return pages

    def _by_title(self, titles: list[str], size: int, **params) -> tuple[dict, dict]:
        finals, pages = {}, {}
        for chunk in batched(titles, size):
            responses = self._query_all(titles="|".join(chunk), redirects=1, **params)
            finals.update(self._final_titles(responses, list(chunk)))
            pages.update(self._pages(responses))
        return finals, pages

    def resolve(self, titles: list[str]) -> dict[str, str | None]:
        """Título canónico; None si no existe o es una página de desambiguación."""
        finals, pages = self._by_title(titles, 50, prop="pageprops", ppprop="disambiguation")
        out = {}
        for title in titles:
            page = pages.get(finals[title], {"missing": True})
            ok = not page.get("missing") and "disambiguation" not in page.get("pageprops", {})
            out[title] = finals[title] if ok else None
        return out

    def extracts(self, titles: list[str]) -> dict[str, str]:
        """Introducción en texto plano; "" si la página no existe."""
        finals, pages = self._by_title(
            titles, 20, prop="extracts", exintro=1, explaintext=1, exlimit=20
        )
        return {t: pages.get(finals[t], {}).get("extract", "") or "" for t in titles}

    def links(self, title: str) -> list[str]:
        """Enlaces salientes a artículos (espacio de nombres 0), en orden y sin repetir."""
        responses = self._query_all(
            titles=title, redirects=1, prop="links", plnamespace=0, pllimit="max"
        )
        seen: dict[str, None] = {}
        for page in self._pages(responses).values():
            for link in page.get("links", []):
                seen.setdefault(link["title"], None)
        return list(seen)

    def langlinks(self, titles: list[str], target: str) -> dict[str, str | None]:
        """Título equivalente en otro idioma; None si no existe."""
        finals, pages = self._by_title(
            titles, 50, prop="langlinks", lllang=target, lllimit="max"
        )
        out = {}
        for title in titles:
            page = pages.get(finals[title], {})
            links = [x for x in page.get("langlinks", []) if x.get("lang") == target]
            out[title] = links[0]["title"] if links and not page.get("missing") else None
        return out

    def search(self, query: str, limit: int = 5) -> list[str]:
        data = self.client.get_json(
            self.url,
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "srnamespace": 0,
            },
        )
        return [x["title"] for x in data.get("query", {}).get("search", [])]
```

Nota: `itertools.batched` existe desde Python 3.12 (el repo exige 3.12).

- [ ] **Step 8: Implementar `openalex.py`**

```python
"""Cliente mínimo de OpenAlex sobre `CachedClient` (paginación por cursor)."""

from research_paths.http import CachedClient

BASE = "https://api.openalex.org"
WORK_FIELDS = (
    "id,title,type,publication_year,language,cited_by_count,referenced_works,"
    "abstract_inverted_index,primary_topic"
)
KEY_URL = "https://openalex.org/settings/api"


def short_id(value: str) -> str:
    """'https://openalex.org/W123' → 'W123'."""
    return value.rsplit("/", 1)[-1]


def abstract_text(inverted: dict | None) -> str:
    """Reconstruye el resumen desde el índice invertido {palabra: [posiciones]}."""
    if not inverted:
        return ""
    positions = {i: word for word, idxs in inverted.items() for i in idxs}
    return " ".join(positions[i] for i in sorted(positions))


class OpenAlex:
    def __init__(self, client: CachedClient, api_key: str | None):
        self.client, self.api_key = client, api_key

    def _paged(self, path: str, params: dict, max_results: int, per_page: int) -> list[dict]:
        out: list[dict] = []
        cursor = "*"
        while cursor and len(out) < max_results:
            page_params = {**params, "per-page": per_page, "cursor": cursor}
            if self.api_key:
                page_params["api_key"] = self.api_key
            data = self.client.get_json(f"{BASE}/{path}", page_params)
            results = data.get("results", [])
            out.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not results:
                break
        return out[:max_results]

    def works(
        self,
        *,
        filter: str,  # noqa: A002 (nombre del parámetro de la API)
        search: str | None = None,
        sort: str | None = None,
        max_results: int = 200,
        per_page: int = 200,
        select: str = WORK_FIELDS,
    ) -> list[dict]:
        params = {"filter": filter, "select": select}
        if search:
            params["search"] = search
        if sort:
            params["sort"] = sort
        return self._paged("works", params, max_results, min(per_page, max_results))

    def topics(self, *, filter: str) -> list[dict]:  # noqa: A002
        params = {"filter": filter, "select": "id,display_name,field,subfield,works_count"}
        return self._paged("topics", params, max_results=10_000, per_page=200)
```

Nota: `A002` no está en las reglas activas de ruff (`select` no incluye `A`); si ruff marca el
`noqa` como innecesario (`RUF100`), quitar los comentarios `# noqa`.

- [ ] **Step 9: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_http.py 02-semestre-2/02-pln/tests/test_wiki.py 02-semestre-2/02-pln/tests/test_openalex.py -q`
Expected: `17 passed`.

- [ ] **Step 10: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): clientes de Wikipedia y OpenAlex con caché en disco y reintentos"
```

---
### Task 5: AL-CPL: descarga, mapeo a Wikipedia ES y particiones sin fuga

**Files:**
- Modify: `M/data/download.py` (reemplazar la plantilla)
- Create: `M/src/research_paths/alcpl.py`
- Create: `M/tests/test_alcpl.py`

**Interfaces:**
- Produces: `DOMAINS: tuple[str, ...]`; `read_pairs(path) -> list[tuple[str, str]]`;
  `load_alcpl(raw_dir: Path) -> pd.DataFrame` (columnas `domain, a, b, label`, en (a, b) b es
  prerrequisito de a); `map_concepts(df, mapping: dict[str, str | None]) -> (pd.DataFrame, dict)`
  (agrega `a_es, b_es`; descarta sin mapeo y colisiones; stats con claves `concepts,
  concepts_unmapped, pairs, pairs_unmapped, pairs_collision, pairs_kept, positives_kept`);
  `Fold(test, val, train)`; `folds() -> list[Fold]`; `split_fold(df, fold) -> (train, val, test)`
  (la identidad de un concepto es `a_es`/`b_es`).

- [ ] **Step 1: Tests**

`M/tests/test_alcpl.py`:

```python
"""Carga de AL-CPL, mapeo a ES y particiones por dominio sin conceptos compartidos."""

import pandas as pd
import pytest

from research_paths.alcpl import DOMAINS, folds, load_alcpl, map_concepts, split_fold


def _write(raw, domain, pairs, preqs):
    (raw / f"{domain}.pairs").write_text("".join(f"{a},{b}\n" for a, b in pairs))
    (raw / f"{domain}.preqs").write_text("".join(f"{a},{b}\n" for a, b in preqs))


@pytest.fixture
def raw(tmp_path):
    # «Media» aparece en dos dominios (como los 40 conceptos compartidos del real).
    _write(tmp_path, "data_mining", [("K-means", "Media"), ("Media", "K-means")], [("K-means", "Media")])
    _write(tmp_path, "geometry", [("Área", "Longitud"), ("Longitud", "Área")], [("Área", "Longitud")])
    _write(tmp_path, "physics", [("Fuerza", "Masa"), ("Masa", "Fuerza")], [("Fuerza", "Masa")])
    _write(tmp_path, "precalculus", [("Varianza", "Media"), ("Media", "Varianza")], [("Varianza", "Media")])
    return tmp_path


def test_load_labels_pairs_from_preqs(raw):
    df = load_alcpl(raw)
    assert list(df.columns) == ["domain", "a", "b", "label"]
    assert len(df) == 8 and df.label.sum() == 4
    row = df[(df.a == "K-means") & (df.b == "Media")].iloc[0]
    assert row.label == 1 and row.domain == "data_mining"


def test_load_rejects_positives_missing_from_pairs(tmp_path):
    for d in DOMAINS:
        _write(tmp_path, d, [("A", "B")], [("A", "B")])
    (tmp_path / "physics.preqs").write_text("A,B\nC,D\n")
    with pytest.raises(ValueError, match="physics"):
        load_alcpl(tmp_path)


def test_map_concepts_drops_unmapped_and_collisions():
    df = pd.DataFrame(
        {
            "domain": ["d"] * 3,
            "a": ["Null_hypothesis", "AdaBoost", "Variance"],
            "b": ["Statistical_hypothesis_testing", "Mean", "Mean"],
            "label": [1, 1, 0],
        }
    )
    mapping = {
        "Null_hypothesis": "Contraste de hipótesis",
        "Statistical_hypothesis_testing": "Contraste de hipótesis",
        "AdaBoost": None,
        "Variance": "Varianza",
        "Mean": "Media",
    }
    out, stats = map_concepts(df, mapping)
    assert out[["a_es", "b_es"]].values.tolist() == [["Varianza", "Media"]]
    assert stats == {
        "concepts": 5,
        "concepts_unmapped": 1,
        "pairs": 3,
        "pairs_unmapped": 1,
        "pairs_collision": 1,
        "pairs_kept": 1,
        "positives_kept": 0,
    }


def test_folds_rotate_test_and_validation():
    fs = folds()
    assert [f.test for f in fs] == list(DOMAINS)
    assert [f.val for f in fs] == [*DOMAINS[1:], DOMAINS[0]]
    for f in fs:
        assert set(f.train) | {f.test, f.val} == set(DOMAINS)


def test_split_has_no_shared_concepts_and_no_inverted_pairs(raw):
    df, _ = map_concepts(load_alcpl(raw), {c: c for c in ["K-means", "Media", "Área", "Longitud",
                                                          "Fuerza", "Masa", "Varianza"]})
    for fold in folds():
        train, val, test = split_fold(df, fold)
        concepts = lambda x: set(x.a_es) | set(x.b_es)  # noqa: E731
        assert not concepts(train) & concepts(test)
        assert not concepts(val) & concepts(test)
        assert not concepts(train) & concepts(val)
        assert len(test) == (df.domain == fold.test).sum()  # test se conserva entero
        pairs = {(a, b) for a, b in zip(train.a_es, train.b_es, strict=True)}
        assert not any((b, a) in pairs for a, b in zip(test.a_es, test.b_es, strict=True))
```

- [ ] **Step 2: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_alcpl.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'research_paths.alcpl'`.

- [ ] **Step 3: Implementar `alcpl.py`**

```python
"""AL-CPL (Liang et al., 2018): prerrequisitos entre conceptos, mapeo a Wikipedia ES y particiones.

En cada par (A, B), B es prerrequisito de A. Los autores ampliaron el conjunto con el par invertido
como negativo y con el cierre transitivo, así que una partición aleatoria de pares filtra la
respuesta: se parte por dominio y se quitan los pares que tocan conceptos del otro lado.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DOMAINS = ("data_mining", "geometry", "physics", "precalculus")


def read_pairs(path: Path) -> list[tuple[str, str]]:
    with Path(path).open(newline="") as f:
        return [(r[0].strip(), r[1].strip()) for r in csv.reader(f) if len(r) >= 2]


def load_alcpl(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        pairs = read_pairs(Path(raw_dir) / f"{domain}.pairs")
        positives = set(read_pairs(Path(raw_dir) / f"{domain}.preqs"))
        if missing := positives - set(pairs):
            raise ValueError(f"{domain}: {len(missing)} positivos no están en .pairs")
        rows += [(domain, a, b, int((a, b) in positives)) for a, b in pairs]
    df = pd.DataFrame(rows, columns=["domain", "a", "b", "label"])
    return df.drop_duplicates(["domain", "a", "b"]).reset_index(drop=True)


def map_concepts(df: pd.DataFrame, mapping: dict[str, str | None]) -> tuple[pd.DataFrame, dict]:
    """Agrega `a_es`, `b_es`; descarta pares sin página ES o cuyos dos conceptos caen en la misma."""
    out = df.assign(a_es=df.a.map(mapping), b_es=df.b.map(mapping))
    unmapped = out.a_es.isna() | out.b_es.isna()
    collision = ~unmapped & (out.a_es == out.b_es)
    keep = ~unmapped & ~collision
    concepts = set(df.a) | set(df.b)
    stats = {
        "concepts": len(concepts),
        "concepts_unmapped": sum(mapping.get(c) is None for c in concepts),
        "pairs": len(df),
        "pairs_unmapped": int(unmapped.sum()),
        "pairs_collision": int(collision.sum()),
        "pairs_kept": int(keep.sum()),
        "positives_kept": int(out.label[keep].sum()),
    }
    return out[keep].reset_index(drop=True), stats


@dataclass(frozen=True)
class Fold:
    test: str
    val: str
    train: tuple[str, ...]


def folds() -> list[Fold]:
    """Test = dominio i, validación = dominio i+1, entrenamiento = los otros dos."""
    out = []
    for i, test in enumerate(DOMAINS):
        val = DOMAINS[(i + 1) % len(DOMAINS)]
        out.append(Fold(test, val, tuple(d for d in DOMAINS if d not in (test, val))))
    return out


def _concepts(df: pd.DataFrame) -> set[str]:
    return set(df.a_es) | set(df.b_es)


def _without(df: pd.DataFrame, concepts: set[str]) -> pd.DataFrame:
    return df[~df.a_es.isin(concepts) & ~df.b_es.isin(concepts)]


def split_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = df[df.domain == fold.test]
    test_c = _concepts(test)
    val = _without(df[df.domain == fold.val], test_c)
    train = _without(df[df.domain.isin(fold.train)], test_c | _concepts(val))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
```

- [ ] **Step 4: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_alcpl.py -q`
Expected: `5 passed`.

- [ ] **Step 5: `download.py`**

Reemplazar `M/data/download.py`:

```python
"""Descarga AL-CPL (Liang et al., 2018) con commit fijo y verifica el SHA-256 de cada archivo.

Pares de conceptos de Wikipedia (inglés) etiquetados como prerrequisito en cuatro dominios: data
mining, geometry, physics y precalculus. Repo: github.com/harrylclc/AL-CPL-dataset (sin licencia
explícita: se descarga para uso de estudio y no se redistribuye). Citar: Liang et al., «Investigating
Active Learning for Concept Prerequisite Learning», EAAI 2018.

Wikipedia y OpenAlex no se descargan aquí: `run.py` los consulta bajo demanda y los guarda en
`data/cache/`.

Uso: uv run python 02-semestre-2/02-pln/data/download.py
"""

import hashlib
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
COMMIT = "11397eaa3ea01c5e06fefcfc1c02a84f21164081"
URL = "https://raw.githubusercontent.com/harrylclc/AL-CPL-dataset/{commit}/data/{name}"
SHA256 = {
    "data_mining.pairs": "9b0d42ac047f7e4b09f77210afcce5b4639f026c608d31f1d49e71c6b718e5da",
    "data_mining.preqs": "8dda7ec59e7199531baad22545d938acb840a20306ed79ae767e82dfb27d62de",
    "geometry.pairs": "883d6fa4622de6c2fc4581b835e3e1d3ab7d1ad7df00728c070897a0d09ec12c",
    "geometry.preqs": "e45787dfe417035c72a2a9cfa984ee7b75193d7b1f27e43a6023b363c9c69c40",
    "physics.pairs": "639070062f36c80e9393367f20f97e22b98ad3a36edbe36a5999ac930ddd13c4",
    "physics.preqs": "e1ce25603a7f3388ca2d7da034f7de7223eb40e62591415525687718fc059643",
    "precalculus.pairs": "20de87fbb394410c400871904d50cb723e5c88d31e12bdcc5919933478ba97cd",
    "precalculus.preqs": "80f8034f323d7853cd1036108292572115ebfdb52de78beeee493a5ca15b3015",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    dest = DATA_DIR / "alcpl"
    dest.mkdir(exist_ok=True)
    for name, digest in SHA256.items():
        path = dest / name
        if path.exists() and _sha256(path) == digest:
            continue
        with urllib.request.urlopen(URL.format(commit=COMMIT, name=name), timeout=60) as r:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(r.read())
        if _sha256(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"SHA-256 de {name} no coincide con el commit {COMMIT[:7]}")
        tmp.replace(path)
    print(f"AL-CPL listo en {dest} ({len(SHA256)} archivos verificados)")


if __name__ == "__main__":
    main()
```

Run: `uv run python 02-semestre-2/02-pln/data/download.py`
Expected: `AL-CPL listo en …/02-pln/data/alcpl (8 archivos verificados)`.

Run: `uv run python -c "from research_paths.alcpl import load_alcpl; from common import data_dir; df = load_alcpl(data_dir('02-semestre-2/02-pln') / 'alcpl'); print(len(df), df.label.sum(), df.groupby('domain').size().to_dict())"`
Expected: `6529 2002 {'data_mining': 826, 'geometry': 1681, 'physics': 1962, 'precalculus': 2060}`
(si hay duplicados, el número baja: anotarlo en el ledger como hecho, no como error).

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): AL-CPL con commit fijo, mapeo a Wikipedia ES y particiones por dominio sin fuga"
```

---

### Task 6: Prerrequisitos: RefD desde cero y embeddings + logística

**Files:**
- Create: `M/src/research_paths/prereq/__init__.py`, `M/src/research_paths/prereq/refd.py`,
  `M/src/research_paths/prereq/features.py`
- Create: `M/tests/test_prereq_refd.py`, `M/tests/test_prereq_features.py`

**Interfaces:**
- Produces: `refd(a, b, links: dict[str, set[str]], concepts: set[str]) -> float` (> 0 ⇒ b es
  prerrequisito de a); `refd_scores(pairs: pd.DataFrame, links, concepts, col_a="a_es",
  col_b="b_es") -> np.ndarray`; `pair_features(ea, eb) -> np.ndarray` (`[a, b, a−b, a·b]`);
  `fit_pair_logreg(X, y, C) -> LogisticRegression`; `select_C(X_tr, y_tr, X_val, y_val,
  grid=(0.01, 0.1, 1.0, 10.0)) -> float`.

- [ ] **Step 1: Tests de RefD**

`M/tests/test_prereq_refd.py`:

```python
"""RefD (Liang et al., 2015) contra un ejemplo calculado a mano."""

import numpy as np
import pandas as pd
import pytest

from research_paths.prereq.refd import refd, refd_scores

LINKS = {"A": {"B", "C", "X"}, "B": set(), "C": {"B"}, "D": {"A", "B"}}
CONCEPTS = {"A", "B", "C", "D"}  # X no es un concepto del conjunto: se ignora


def test_refd_hand_example():
    # A enlaza a {B, C}; ambos «llegan» a B (B mismo y C → B): 2/2. B no enlaza a nada: 0.
    assert refd("A", "B", LINKS, CONCEPTS) == pytest.approx(1.0)
    assert refd("B", "A", LINKS, CONCEPTS) == pytest.approx(-1.0)
    # D enlaza a {A, B}: A llega a C y B no → 1/2. C enlaza a {B}, que no llega a D → 0.
    assert refd("D", "C", LINKS, CONCEPTS) == pytest.approx(0.5)


def test_refd_is_antisymmetric():
    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(8)]
    links = {n: {m for m in names if m != n and rng.random() < 0.4} for n in names}
    for a in names:
        for b in names:
            assert refd(a, b, links, set(names)) == pytest.approx(-refd(b, a, links, set(names)))


def test_refd_scores_over_dataframe():
    pairs = pd.DataFrame({"a_es": ["A", "B"], "b_es": ["B", "A"]})
    np.testing.assert_allclose(refd_scores(pairs, LINKS, CONCEPTS), [1.0, -1.0])
```

- [ ] **Step 2: Tests de embeddings + logística**

`M/tests/test_prereq_features.py`:

```python
"""Rasgos de par asimétricos y logística con C elegido en validación."""

import numpy as np

from research_paths.evaluation import average_precision
from research_paths.prereq.features import fit_pair_logreg, pair_features, select_C


def _toy(rng, n_concepts=60, n_pairs=600):
    # Cada concepto tiene un «nivel» en la primera coordenada; b es prerrequisito de a si es de
    # nivel menor. Un modelo simétrico no puede aprenderlo; [a, b, a−b, a·b] sí.
    emb = rng.normal(size=(n_concepts, 8))
    level = emb[:, 0]
    i, j = rng.integers(0, n_concepts, (2, n_pairs))
    keep = i != j
    i, j = i[keep], j[keep]
    return pair_features(emb[i], emb[j]), (level[j] < level[i]).astype(int)


def test_pair_features_are_asymmetric():
    a, b = np.array([[1.0, 2.0]]), np.array([[3.0, 5.0]])
    f = pair_features(a, b)
    assert f.shape == (1, 8)
    assert not np.allclose(f, pair_features(b, a))
    np.testing.assert_allclose(f[0, 4:6], [-2.0, -3.0])


def test_logreg_learns_direction_on_unseen_pairs():
    rng = np.random.default_rng(0)
    X_tr, y_tr = _toy(rng)
    X_val, y_val = _toy(rng)
    C = select_C(X_tr, y_tr, X_val, y_val)
    model = fit_pair_logreg(X_tr, y_tr, C)
    assert average_precision(y_val, model.predict_proba(X_val)[:, 1]) > 0.9
```

- [ ] **Step 3: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_prereq_refd.py 02-semestre-2/02-pln/tests/test_prereq_features.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'research_paths.prereq'`.

- [ ] **Step 4: Implementar**

`M/src/research_paths/prereq/__init__.py`:

```python
"""Modelos de prerrequisito entre conceptos: RefD, embeddings + logística, cross-encoder y LLM."""
```

`M/src/research_paths/prereq/refd.py`:

```python
"""RefD (Liang et al., 2015): ¿A «se refiere» a B más de lo que B se refiere a A?

RefD(A, B) = Σ_c w(c, A)·r(c, B) / Σ_c w(c, A) − Σ_c w(c, B)·r(c, A) / Σ_c w(c, B), con
w(c, X) = 1 si X enlaza a c y r(c, Y) = 1 si c enlaza a Y o c = Y. Los c se limitan a `concepts`
(solo se descargan los enlaces de esos conceptos, no de toda Wikipedia).
"""

import numpy as np
import pandas as pd


def _refers(x: str, y: str, links: dict[str, set[str]], concepts: set[str]) -> float:
    outgoing = links.get(x, set()) & concepts
    if not outgoing:
        return 0.0
    hits = sum(1 for c in outgoing if c == y or y in links.get(c, set()))
    return hits / len(outgoing)


def refd(a: str, b: str, links: dict[str, set[str]], concepts: set[str]) -> float:
    """> 0: b se parece más a un prerrequisito de a que al revés."""
    return _refers(a, b, links, concepts) - _refers(b, a, links, concepts)


def refd_scores(
    pairs: pd.DataFrame,
    links: dict[str, set[str]],
    concepts: set[str],
    col_a: str = "a_es",
    col_b: str = "b_es",
) -> np.ndarray:
    return np.array(
        [refd(a, b, links, concepts) for a, b in zip(pairs[col_a], pairs[col_b], strict=True)]
    )
```

Nota sobre el tercer caso del test: D enlaza a {A, B}; para `refd("D", "C")`: c = A llega a C
(A → C) y c = B no (B no enlaza a nada) → 1/2; C enlaza a {B}, que no llega a D → 0; RefD = 0,5.

`M/src/research_paths/prereq/features.py`:

```python
"""Embeddings de concepto → rasgos de par asimétricos → regresión logística."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from research_paths.evaluation import average_precision


def pair_features(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """[a, b, a − b, a·b]: el orden del par importa (a−b cambia de signo al invertirlo)."""
    return np.hstack([ea, eb, ea - eb, ea * eb])


def fit_pair_logreg(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(X, y)


def select_C(X_tr, y_tr, X_val, y_val, grid=(0.01, 0.1, 1.0, 10.0)) -> float:
    """C con mayor AP en validación (en empate, el menor: más regularizado)."""
    scores = [
        average_precision(y_val, fit_pair_logreg(X_tr, y_tr, c).predict_proba(X_val)[:, 1])
        for c in grid
    ]
    return float(grid[int(np.argmax(scores))])
```

- [ ] **Step 5: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_prereq_refd.py 02-semestre-2/02-pln/tests/test_prereq_features.py -q`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): RefD desde cero y embeddings + logística para prerrequisitos"
```

---

### Task 7: Clasificador de texto afinado (XLM-R: pares o un texto)

**Files:**
- Create: `M/src/research_paths/textclf.py`
- Create: `M/tests/test_textclf.py`

**Interfaces:**
- Consumes: `average_precision` (Task 3).
- Produces: `TrainConfig(model_name="xlm-roberta-base", max_length=256, epochs=2, batch_size=16,
  lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, seed=42)`; `default_device() -> str`;
  `tiny_model(texts, vocab_size=200, num_labels=2, max_length=64) -> (model, tokenizer)` (sin
  descargas); `load_pretrained(name, num_labels=2) -> (model, tokenizer)`;
  `train_classifier(texts_a, labels, val_a, val_labels, cfg, *, texts_b=None, val_b=None,
  model=None, tokenizer=None, device="cpu", log=print) -> (model, tokenizer, history)` donde
  `history = [{"epoch", "loss", "val_ap"}]` y el modelo devuelto es el de la mejor época;
  `predict_proba(model, tokenizer, texts_a, texts_b=None, *, max_length=256, batch_size=32,
  device="cpu") -> np.ndarray`; `save_classifier(model, tokenizer, path)`;
  `load_classifier(path) -> (model, tokenizer)`.

- [ ] **Step 1: Tests**

`M/tests/test_textclf.py`:

```python
"""Ajuste de un XLM-R diminuto (sin descargas): aprende, elige la mejor época y se guarda."""

import numpy as np
import pytest

pytest.importorskip("transformers")

from research_paths.textclf import (  # noqa: E402
    TrainConfig,
    load_classifier,
    predict_proba,
    save_classifier,
    tiny_model,
    train_classifier,
)

WORDS = "gato perro casa árbol río sol luna mar".split()


def _pairs(rng, n):
    a = [" ".join(rng.choice(WORDS, 4)) for _ in range(n)]
    y = rng.integers(0, 2, n)
    b = [("básico elemental " if t else "avanzado complejo ") + " ".join(rng.choice(WORDS, 3)) for t in y]
    return a, b, y


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return _pairs(rng, 160), _pairs(rng, 60)


CFG = TrainConfig(model_name="tiny", max_length=64, epochs=8, batch_size=16, lr=3e-3, warmup_ratio=0.0)


def test_pair_classifier_learns_and_keeps_best_epoch(data):
    (a, b, y), (va, vb, vy) = data
    model, tok = tiny_model(a + b)
    model, tok, history = train_classifier(
        a, y, va, vy, CFG, texts_b=b, val_b=vb, model=model, tokenizer=tok, log=lambda *_: None
    )
    assert len(history) == 8
    best = max(h["val_ap"] for h in history)
    assert best >= 0.95
    p = predict_proba(model, tok, va, vb, max_length=64)
    from research_paths.evaluation import average_precision

    assert average_precision(vy, p) == pytest.approx(best)


def test_single_text_mode_and_probabilities_in_range(data):
    (a, b, y), (va, vb, vy) = data
    texts = [x + " " + z for x, z in zip(a, b, strict=True)]
    val = [x + " " + z for x, z in zip(va, vb, strict=True)]
    model, tok = tiny_model(texts)
    cfg = TrainConfig(model_name="tiny", max_length=64, epochs=2, batch_size=16, lr=3e-3)
    model, tok, _ = train_classifier(texts, y, val, vy, cfg, model=model, tokenizer=tok, log=lambda *_: None)
    p = predict_proba(model, tok, val, max_length=64)
    assert p.shape == (60,) and np.all((p >= 0) & (p <= 1))


def test_save_and_load_give_identical_predictions(data, tmp_path):
    (a, b, y), (va, vb, _) = data
    model, tok = tiny_model(a + b)
    before = predict_proba(model, tok, va, vb, max_length=64)
    save_classifier(model, tok, tmp_path / "m")
    model2, tok2 = load_classifier(tmp_path / "m")
    np.testing.assert_allclose(predict_proba(model2, tok2, va, vb, max_length=64), before, atol=1e-6)
```

- [ ] **Step 2: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_textclf.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'research_paths.textclf'`.

- [ ] **Step 3: Implementar**

`M/src/research_paths/textclf.py`:

```python
"""Clasificador de texto afinado (XLM-R): un texto o un par (cross-encoder).

`tiny_model` construye un XLM-R diminuto con un tokenizador BPE entrenado en local: sirve para
tests y notebooks sin descargar nada. Para el proyecto se usa `load_pretrained("xlm-roberta-base")`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research_paths.evaluation import average_precision


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 256  # el par completo; se trunca `longest_first`
    epochs: int = 2
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def tiny_model(texts: list[str], vocab_size: int = 200, num_labels: int = 2, max_length: int = 64):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import (
        PreTrainedTokenizerFast,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
    )

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    special = ["<s>", "<pad>", "</s>", "<unk>"]
    tk.train_from_iterator(
        texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special, show_progress=False)
    )
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=max_length,
    )
    torch.manual_seed(0)
    config = XLMRobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=max_length + 16,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
    )
    return XLMRobertaForSequenceClassification(config), tokenizer


def load_pretrained(name: str, num_labels: int = 2):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    return model, tokenizer


def _encode(tokenizer, texts_a, texts_b, max_length: int, device: str):
    enc = tokenizer(
        list(texts_a),
        list(texts_b) if texts_b is not None else None,
        truncation="longest_first",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def predict_proba(
    model, tokenizer, texts_a, texts_b=None, *, max_length: int = 256, batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """P(clase 1) para cada texto o par."""
    model.to(device).eval()
    out = []
    for s in range(0, len(texts_a), batch_size):
        b = texts_b[s : s + batch_size] if texts_b is not None else None
        logits = model(**_encode(tokenizer, texts_a[s : s + batch_size], b, max_length, device)).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_classifier(
    texts_a, labels, val_a, val_labels, cfg: TrainConfig, *, texts_b=None, val_b=None,
    model=None, tokenizer=None, device: str = "cpu", log=print,
):
    """AdamW + calentamiento lineal, pérdida con pesos por clase; devuelve la mejor época (AP val)."""
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if model is None:
        model, tokenizer = load_pretrained(cfg.model_name)
    model.to(device)
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2).astype(float)
    weight = torch.tensor(len(labels) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * int(np.ceil(len(texts_a) / cfg.batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    history, best_state, best_ap = [], None, -1.0
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(texts_a))
        losses = []
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s : s + cfg.batch_size]
            b = [texts_b[i] for i in idx] if texts_b is not None else None
            enc = _encode(tokenizer, [texts_a[i] for i in idx], b, cfg.max_length, device)
            y = torch.tensor(labels[idx], device=device)
            loss = F.cross_entropy(model(**enc).logits.float(), y, weight=weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(float(loss))
        p = predict_proba(model, tokenizer, val_a, val_b, max_length=cfg.max_length, device=device)
        ap = average_precision(val_labels, p)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_ap": ap})
        log(f"  época {epoch + 1}: pérdida {np.mean(losses):.4f} · AP val {ap:.3f}")
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tokenizer, history


def save_classifier(model, tokenizer, path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def load_classifier(path: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return (
        AutoModelForSequenceClassification.from_pretrained(path),
        AutoTokenizer.from_pretrained(path),
    )
```

- [ ] **Step 4: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_textclf.py -q`
Expected: `3 passed` en pocos segundos.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): clasificador de texto afinado (XLM-R, pares o un texto) con mejor época por AP"
```

---

### Task 8: Juez LLM de prerrequisitos (Claude) con caché y reintentos

**Files:**
- Create: `M/src/research_paths/prereq/llm.py`
- Create: `M/tests/test_prereq_llm.py`

**Interfaces:**
- Consumes: `DiskCache`, `require_env`, `MissingKeyError` (Task 4).
- Produces: `LLMJudge(cache_dir, model="claude-haiku-5-5", client=None, max_workers=8,
  max_retries=6, base_delay=2.0, sleep=time.sleep, max_chars=1200)` con
  `.judge(items: list[tuple[str, str, str, str]]) -> list[bool | None]` (cada item = título A, texto
  A, título B, texto B; None = respuesta no interpretable) y `.stats = {"api", "cache", "unparsed"}`;
  `build_prompt(title_a, text_a, title_b, text_b, max_chars) -> str`.

- [ ] **Step 1: Tests**

`M/tests/test_prereq_llm.py`:

```python
"""Juez LLM con un cliente falso: herramienta forzada, caché, reintentos y respuestas raras."""

from types import SimpleNamespace

import pytest

from research_paths.http import MissingKeyError
from research_paths.prereq.llm import LLMJudge, build_prompt


class Overloaded(Exception):
    status_code = 529


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)  # cada elemento: bool | "raro" | Exception
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if item == "raro":
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="no sé")])
        block = SimpleNamespace(
            type="tool_use", name="responder", input={"prerrequisito": item, "razon": "porque"}
        )
        return SimpleNamespace(content=[block])


def _judge(tmp_path, script, **kw):
    fake = SimpleNamespace(messages=FakeMessages(script))
    return LLMJudge(tmp_path, client=fake, max_workers=1, sleep=lambda s: None, **kw), fake


ITEM = ("Varianza", "La varianza mide la dispersión.", "Media", "La media es el promedio.")


def test_prompt_contains_both_concepts_and_truncates():
    p = build_prompt("A", "x" * 5000, "B", "texto B", max_chars=100)
    assert "A" in p and "B" in p and "texto B" in p and "x" * 101 not in p


def test_forces_the_tool_and_parses_answers(tmp_path):
    judge, fake = _judge(tmp_path, [True, False])
    other = ("Media", "La media es el promedio.", "Varianza", "La varianza mide la dispersión.")
    assert judge.judge([ITEM, other]) == [True, False]
    call = fake.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "responder"}
    assert call["model"] == "claude-haiku-5-5"


def test_cache_avoids_paying_twice(tmp_path):
    judge, fake = _judge(tmp_path, [True])
    judge.judge([ITEM])
    judge2, fake2 = _judge(tmp_path, [])
    assert judge2.judge([ITEM]) == [True]
    assert fake2.messages.calls == [] and judge2.stats["cache"] == 1


def test_unparseable_answer_is_none_and_counted(tmp_path):
    judge, _ = _judge(tmp_path, ["raro"])
    assert judge.judge([ITEM]) == [None]
    assert judge.stats["unparsed"] == 1


def test_retries_overload_then_succeeds(tmp_path):
    judge, fake = _judge(tmp_path, [Overloaded(), Overloaded(), True])
    assert judge.judge([ITEM]) == [True]
    assert len(fake.messages.calls) == 3


def test_non_retryable_error_propagates(tmp_path):
    judge, _ = _judge(tmp_path, [ValueError("mal")])
    with pytest.raises(ValueError):
        judge.judge([ITEM])


def test_missing_key_only_matters_when_the_api_is_needed(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge, _ = _judge(tmp_path, [True])
    judge.judge([ITEM])  # llena la caché con el cliente falso
    cached_only = LLMJudge(tmp_path, client=None, max_workers=1)
    assert cached_only.judge([ITEM]) == [True]  # sin clave, pero todo está en caché
    with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY"):
        cached_only.judge([("X", "x", "Y", "y")])
```

- [ ] **Step 2: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_prereq_llm.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'research_paths.prereq.llm'`.

- [ ] **Step 3: Implementar**

`M/src/research_paths/prereq/llm.py`:

```python
"""Juez LLM (Claude) para prerrequisitos: herramienta forzada, caché por respuesta y reintentos."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from research_paths.http import DiskCache, require_env

SYSTEM = (
    "Eres un docente experto en diseñar rutas de aprendizaje. Decides si un concepto es "
    "prerrequisito de otro: si hay que entender B para poder entender A."
)
TOOL = {
    "name": "responder",
    "description": "Registra si B es prerrequisito de A.",
    "input_schema": {
        "type": "object",
        "properties": {
            "prerrequisito": {"type": "boolean", "description": "true si B es prerrequisito de A"},
            "razon": {"type": "string", "description": "una frase"},
        },
        "required": ["prerrequisito", "razon"],
    },
}
RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
KEY_URL = "https://console.anthropic.com/settings/keys"


def build_prompt(title_a: str, text_a: str, title_b: str, text_b: str, max_chars: int) -> str:
    return (
        f"Concepto A: {title_a}\n{text_a[:max_chars]}\n\n"
        f"Concepto B: {title_b}\n{text_b[:max_chars]}\n\n"
        "¿Hay que entender B antes de poder entender A?"
    )


def _retryable(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if status in RETRY_STATUS:
        return True
    name = type(err).__name__
    return name in {"APIConnectionError", "APITimeoutError"}


class LLMJudge:
    def __init__(
        self,
        cache_dir: Path,
        model: str = "claude-haiku-5-5",
        client=None,
        max_workers: int = 8,
        max_retries: int = 6,
        base_delay: float = 2.0,
        sleep=time.sleep,
        max_chars: int = 1200,
    ):
        self.cache = DiskCache(cache_dir)
        self.model, self._client = model, client
        self.max_workers, self.max_retries, self.base_delay = max_workers, max_retries, base_delay
        self.sleep, self.max_chars = sleep, max_chars
        self.stats = {"api": 0, "cache": 0, "unparsed": 0}
        self._lock = threading.Lock()

    @property
    def client(self):
        if self._client is None:
            require_env("ANTHROPIC_API_KEY", KEY_URL)
            import anthropic

            self._client = anthropic.Anthropic(max_retries=0)  # los reintentos son nuestros
        return self._client

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _call(self, prompt: str):
        for attempt in range(self.max_retries):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=0,
                    system=SYSTEM,
                    tools=[TOOL],
                    tool_choice={"type": "tool", "name": "responder"},
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as err:
                if not _retryable(err) or attempt == self.max_retries - 1:
                    raise
                self.sleep(self.base_delay * 2**attempt)
        raise AssertionError("inalcanzable")

    @staticmethod
    def _parse(response) -> tuple[bool | None, str]:
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == "responder":
                value = block.input.get("prerrequisito")
                if isinstance(value, bool):
                    return value, str(block.input.get("razon", ""))
        return None, ""

    def _ask(self, item: tuple[str, str, str, str]) -> bool | None:
        prompt = build_prompt(*item, max_chars=self.max_chars)
        key = f"{self.model}\n{SYSTEM}\n{prompt}"
        hit = self.cache.get(key)
        if hit is not None:
            self._count("cache")
            answer = hit["answer"]
        else:
            answer, reason = self._parse(self._call(prompt))
            self._count("api")
            self.cache.set(key, {"answer": answer, "reason": reason})
        if answer is None:
            self._count("unparsed")
        return answer

    def judge(self, items: list[tuple[str, str, str, str]]) -> list[bool | None]:
        if self.max_workers <= 1:
            return [self._ask(item) for item in items]
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._ask, items))
```

Si la API rechaza `temperature` para `claude-haiku-5-5` (error 400 al correr la Tarea 14), quitar
ese argumento y registrar `Ruling` en el ledger; los tests no dependen de él.

- [ ] **Step 4: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_prereq_llm.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): juez LLM de prerrequisitos con herramienta forzada, caché y reintentos"
```

---
### Task 9: Lecturas: temas de evaluación, candidatos y orden «empieza aquí»

**Files:**
- Create: `M/src/research_paths/readings.py`
- Modify: `M/tests/conftest.py` (agregar `FakeOpenAlex`, `make_work` y la fixture `oa_world`)
- Create: `M/tests/test_readings.py`

**Interfaces:**
- Consumes: `abstract_text`, `short_id` (Task 4); `BM25`, `simple_tokenize`, `cosine_scores`,
  `Encoder` (Task 3); `recall_at_k`, `ndcg_at_k` (Task 3).
- Produces: `EVAL_FIELDS = ("17", "26", "31", "11")`; `ORDERS = ("openalex", "citas", "bm25",
  "denso", "combinado")`; `EvalTopic(topic_id, name, field, review_id, review_year, gold)`;
  `works_frame(works) -> pd.DataFrame` (columnas `id, title, abstract, text, year, citations, type,
  language, relevance_rank`); `sample_eval_topics(oa, fields=EVAL_FIELDS, per_field=15, seed=42)
  -> list[EvalTopic]`; `topic_candidates(oa, topic, n=200) -> pd.DataFrame`;
  `intro_training_set(oa, eval_topics, fields=EVAL_FIELDS, topics_per_field=25, per_topic=20,
  seed=43) -> pd.DataFrame` (+ `label`, `topic_id`); `zscore(x)`; `start_here_score(relevance,
  p_intro, citations) -> np.ndarray`; `rank_ids(ids, scores) -> list[str]`;
  `ranking_orders(cands, query, encoder, p_intro) -> dict[str, list[str]]`;
  `topic_metrics(cands, gold, orders, k=20) -> dict` (`ceiling, abstract_frac, n_candidates,
  orders: {nombre: {recall, ndcg}}`).
- Produces (tests): fixture `oa_world` → `FakeOpenAlex` con 2 campos × 3 temas (uno sin revisión).

- [ ] **Step 1: OpenAlex falso en `conftest.py`**

Agregar al final de `M/tests/conftest.py`:

```python
class FakeOpenAlex:
    """OpenAlex en memoria; entiende los filtros que usa `readings` (y falla con los demás)."""

    def __init__(self, topics: dict[str, list[dict]], works: list[dict]):
        self.topic_list, self.all_works = topics, works
        self.calls: list[tuple[str, str | None]] = []

    def topics(self, *, filter):  # noqa: A002
        return list(self.topic_list.get(filter.split(":", 1)[1], []))

    def works(self, *, filter, search=None, sort=None, max_results=200, per_page=200, select=None):  # noqa: A002
        self.calls.append((filter, search))
        out = [w for w in self.all_works if _matches(w, filter)]
        if sort == "cited_by_count:desc":
            out.sort(key=lambda w: -(w.get("cited_by_count") or 0))
        return out[:max_results]


def _matches(work: dict, filt: str) -> bool:
    for cond in filt.split(","):
        key, val = cond.split(":", 1)
        if key == "primary_topic.id":
            ok = work["topic"] == val
        elif key == "type":
            ok = work["type"] in val.split("|")
        elif key == "publication_year":
            year = work["publication_year"] or 0
            if val.startswith("<"):
                ok = year < int(val[1:])
            else:
                lo, hi = map(int, val.split("-"))
                ok = lo <= year <= hi
        elif key == "referenced_works_count":
            ok = len(work.get("referenced_works") or []) > int(val[1:])
        elif key == "has_abstract":
            ok = bool(work.get("abstract_inverted_index"))
        else:
            raise ValueError(f"filtro no soportado por el falso: {key}")
        if not ok:
            return False
    return True


def make_work(wid, topic, kind, year, cites, abstract="", refs=()):
    return {
        "id": f"https://openalex.org/{wid}",
        "title": f"Trabajo {wid}",
        "type": kind,
        "publication_year": year,
        "language": "en",
        "cited_by_count": cites,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
        "abstract_inverted_index": {w: [i] for i, w in enumerate(abstract.split())} or None,
        "primary_topic": {"id": f"https://openalex.org/{topic}"},
        "topic": topic,
    }


INTRO_WORDS = "revisión panorama introducción tutorial conceptos básicos"
SPECIFIC_WORDS = "proponemos método experimento resultados mejora específica"


def build_oa_world() -> FakeOpenAlex:
    """2 campos × 3 temas. T3 y T6 no tienen revisión válida. En cada tema con revisión, sus 25
    referencias son los 25 trabajos más citados (así «citas» y «combinado» las encuentran)."""
    topics = {
        "17": [{"id": f"https://openalex.org/T{i}", "display_name": f"Tema {i}"} for i in (1, 2, 3)],
        "26": [{"id": f"https://openalex.org/T{i}", "display_name": f"Tema {i}"} for i in (4, 5, 6)],
    }
    works = []
    for t in range(1, 7):
        tid = f"T{t}"
        earlier = []
        for k in range(40):
            wid = f"W{t}{k:02d}"
            is_review = k % 4 == 0
            words = INTRO_WORDS if is_review else SPECIFIC_WORDS
            works.append(
                make_work(
                    wid, tid, "review" if is_review else "article", 2000 + k % 19,
                    cites=1000 - 10 * k, abstract=f"{words} tema {t}" if k % 3 else "",
                )
            )
            earlier.append(wid)
        if t not in (3, 6):
            works.append(make_work(f"R{t}", tid, "review", 2020, 5000, refs=earlier[:25]))
    return FakeOpenAlex(topics, works)


@pytest.fixture
def oa_world():
    return build_oa_world()
```

- [ ] **Step 2: Tests**

`M/tests/test_readings.py`:

```python
"""Lecturas: temas con revisión, candidatos sin futuro, conjunto de entrenamiento y órdenes."""

import numpy as np
import pytest

from research_paths.readings import (
    ORDERS,
    EvalTopic,
    intro_training_set,
    ranking_orders,
    sample_eval_topics,
    start_here_score,
    topic_candidates,
    topic_metrics,
    works_frame,
    zscore,
)
from research_paths.retrieval import hashing_encoder


def test_works_frame_tolerates_null_fields():
    df = works_frame(
        [{"id": None, "title": None, "abstract_inverted_index": None, "publication_year": None,
          "cited_by_count": None, "type": None}]
    )  # fmt: skip
    row = df.iloc[0]
    assert (row.title, row.abstract, row.text, row.year, row.citations) == ("", "", "", 0, 0)


def test_sample_eval_topics_skips_topics_without_review_and_is_reproducible(oa_world):
    topics = sample_eval_topics(oa_world, fields=("17", "26"), per_field=5, seed=42)
    assert sorted(t.topic_id for t in topics) == ["T1", "T2", "T4", "T5"]
    t1 = next(t for t in topics if t.topic_id == "T1")
    assert t1.review_id == "R1" and t1.review_year == 2020 and len(t1.gold) == 25
    assert topics == sample_eval_topics(oa_world, fields=("17", "26"), per_field=5, seed=42)
    assert len(sample_eval_topics(oa_world, fields=("17",), per_field=1, seed=42)) == 1


def test_candidates_exclude_the_review_and_the_future(oa_world):
    topic = EvalTopic("T1", "Tema 1", "17", "R1", 2010, frozenset({"W100"}))
    cands = topic_candidates(oa_world, topic)
    assert len(cands) > 0 and cands.year.max() < 2010 and "R1" not in set(cands.id)
    assert list(cands.relevance_rank) == list(range(len(cands)))


def test_intro_training_set_is_balanced_and_excludes_evaluation(oa_world):
    evals = sample_eval_topics(oa_world, fields=("17", "26"), per_field=1, seed=42)
    df = intro_training_set(oa_world, evals, fields=("17", "26"), topics_per_field=3, per_topic=5)
    assert not set(df.topic_id) & {t.topic_id for t in evals}
    gold = set().union(*(t.gold for t in evals))
    assert not set(df.id) & gold
    assert (df.abstract != "").all()
    counts = df.groupby(["topic_id", "label"]).size().unstack()
    assert (counts[0] == counts[1]).all()


def test_start_here_score_is_sum_of_zscores():
    rel, p, cites = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]), np.array([0, 9, 99])
    expected = zscore(rel) + zscore(np.log1p(cites))
    np.testing.assert_allclose(start_here_score(rel, p, cites), expected)
    assert np.all(zscore(np.ones(3)) == 0)


def test_ranking_orders(oa_world):
    topic = EvalTopic("T1", "Tema 1", "17", "R1", 2020, frozenset())
    cands = topic_candidates(oa_world, topic)
    p = np.linspace(0, 1, len(cands))
    orders = ranking_orders(cands, "tema 1 revisión", hashing_encoder(), p)
    assert set(orders) == set(ORDERS)
    assert orders["openalex"] == list(cands.id)
    cites = dict(zip(cands.id, cands.citations, strict=True))
    assert [cites[i] for i in orders["citas"]] == sorted(cands.citations, reverse=True)
    for ranked in orders.values():
        assert sorted(ranked) == sorted(cands.id)


def test_ranking_orders_and_metrics_with_no_candidates():
    empty = works_frame([])
    orders = ranking_orders(empty, "x", hashing_encoder(), np.zeros(0))
    m = topic_metrics(empty, frozenset({"W1"}), orders)
    assert m["ceiling"] == 0 and m["abstract_frac"] == 0
    assert all(v == {"recall": 0.0, "ndcg": 0.0} for v in m["orders"].values())


def test_topic_metrics_ceiling(oa_world):
    topic = EvalTopic("T1", "Tema 1", "17", "R1", 2020, frozenset({"W100", "W101", "NO_ESTA"}))
    cands = topic_candidates(oa_world, topic)
    orders = ranking_orders(cands, "tema", hashing_encoder(), np.zeros(len(cands)))
    m = topic_metrics(cands, topic.gold, orders)
    assert m["ceiling"] == pytest.approx(2 / 3)
    assert m["orders"]["citas"]["recall"] == pytest.approx(2 / 3)  # W100 y W101 son los más citados
```

- [ ] **Step 3: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_readings.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'research_paths.readings'`.

- [ ] **Step 4: Implementar**

`M/src/research_paths/readings.py`:

```python
"""Lecturas: temas de evaluación en OpenAlex, candidatos y orden «empieza aquí».

La capa 2 se evalúa sobre temas de investigación de OpenAlex: en cada tema, las referencias de su
revisión más citada son la lista de referencia, y los candidatos son trabajos del mismo tema
publicados antes de esa revisión (sin «ver el futuro»).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_paths.evaluation import ndcg_at_k, recall_at_k
from research_paths.openalex import abstract_text, short_id
from research_paths.retrieval import BM25, Encoder, cosine_scores, simple_tokenize

# Computer Science, Mathematics, Physics and Astronomy, Agricultural and Biological Sciences.
EVAL_FIELDS = ("17", "26", "31", "11")
REVIEW_FILTER = "type:review,publication_year:2010-2024,referenced_works_count:>19"
ORDERS = ("openalex", "citas", "bm25", "denso", "combinado")
COLUMNS = ["id", "title", "abstract", "text", "year", "citations", "type", "language", "relevance_rank"]


@dataclass(frozen=True)
class EvalTopic:
    topic_id: str
    name: str
    field: str
    review_id: str
    review_year: int
    gold: frozenset[str]


def works_frame(works: list[dict]) -> pd.DataFrame:
    """Trabajos de OpenAlex → columnas fijas; los nulos pasan a "" o 0."""
    rows = []
    for rank, w in enumerate(works):
        title = (w.get("title") or "").strip()
        abstract = abstract_text(w.get("abstract_inverted_index"))
        rows.append(
            {
                "id": short_id(w.get("id") or ""),
                "title": title,
                "abstract": abstract,
                "text": f"{title}. {abstract}" if abstract else title,
                "year": int(w.get("publication_year") or 0),
                "citations": int(w.get("cited_by_count") or 0),
                "type": w.get("type") or "",
                "language": w.get("language") or "",
                "relevance_rank": rank,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _shuffled_topics(oa, field: str, rng: np.random.Generator) -> list[dict]:
    topics = sorted(oa.topics(filter=f"field.id:{field}"), key=lambda t: t["id"])
    return [topics[i] for i in rng.permutation(len(topics))]


def sample_eval_topics(oa, fields=EVAL_FIELDS, per_field: int = 15, seed: int = 42) -> list[EvalTopic]:
    """Por campo, los primeros `per_field` temas (barajados con semilla) con una revisión válida."""
    rng = np.random.default_rng(seed)
    out = []
    for field in fields:
        found = 0
        for topic in _shuffled_topics(oa, field, rng):
            if found == per_field:
                break
            tid = short_id(topic["id"])
            reviews = oa.works(
                filter=f"primary_topic.id:{tid},{REVIEW_FILTER}",
                sort="cited_by_count:desc",
                max_results=1,
                select="id,publication_year,referenced_works",
            )
            if not reviews:
                continue
            review = reviews[0]
            gold = frozenset(short_id(x) for x in review.get("referenced_works") or [])
            if not gold or not review.get("publication_year"):
                continue
            name = topic.get("display_name") or tid
            year = int(review["publication_year"])
            out.append(EvalTopic(tid, name, field, short_id(review["id"]), year, gold))
            found += 1
    return out


def topic_candidates(oa, topic: EvalTopic, n: int = 200) -> pd.DataFrame:
    works = oa.works(
        filter=f"primary_topic.id:{topic.topic_id},publication_year:<{topic.review_year}",
        search=topic.name,
        max_results=n,
    )
    df = works_frame(works)
    df = df[df.id != topic.review_id].reset_index(drop=True)
    df["relevance_rank"] = np.arange(len(df))
    return df


def intro_training_set(
    oa, eval_topics: list[EvalTopic], fields=EVAL_FIELDS, topics_per_field: int = 25,
    per_topic: int = 20, seed: int = 43,
) -> pd.DataFrame:
    """Etiqueta débil: `review` = 1, `article` = 0, balanceada por tema, de temas NO evaluados y
    sin las revisiones de referencia ni sus referencias."""
    rng = np.random.default_rng(seed)
    skip_topics = {t.topic_id for t in eval_topics}
    skip_works = set().union(*(t.gold | {t.review_id} for t in eval_topics))
    frames = []
    for field in fields:
        used = 0
        for topic in _shuffled_topics(oa, field, rng):
            if used == topics_per_field:
                break
            tid = short_id(topic["id"])
            if tid in skip_topics:
                continue
            by_kind = {}
            for kind in ("review", "article"):
                w = works_frame(
                    oa.works(
                        filter=f"primary_topic.id:{tid},type:{kind},has_abstract:true",
                        sort="cited_by_count:desc",
                        max_results=per_topic,
                    )
                )
                by_kind[kind] = w[~w.id.isin(skip_works) & (w.abstract != "")]
            n = min(len(by_kind["review"]), len(by_kind["article"]))
            if n == 0:
                continue
            for kind, label in (("review", 1), ("article", 0)):
                frames.append(by_kind[kind].head(n).assign(label=label, topic_id=tid))
            used += 1
    if not frames:
        return pd.DataFrame(columns=[*COLUMNS, "label", "topic_id"])
    return pd.concat(frames, ignore_index=True)


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def start_here_score(relevance, p_intro, citations) -> np.ndarray:
    """«Empieza aquí» = z(relevancia) + z(P(introductorio)) + z(log(1 + citas)), estandarizadas
    dentro de los candidatos. Pesos iguales, fijados antes de ver test (spec §4.2)."""
    return (
        zscore(relevance)
        + zscore(p_intro)
        + zscore(np.log1p(np.asarray(citations, dtype=float)))
    )


def rank_ids(ids: list[str], scores) -> list[str]:
    """Orden descendente estable (en empate se respeta el orden de OpenAlex)."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    return [ids[i] for i in order]


def ranking_orders(cands: pd.DataFrame, query: str, encoder: Encoder, p_intro) -> dict[str, list[str]]:
    ids = cands.id.tolist()
    if not ids:
        return {name: [] for name in ORDERS}
    texts = cands.text.tolist()
    bm25 = BM25().fit([simple_tokenize(t) for t in texts]).scores(simple_tokenize(query))
    dense = cosine_scores(encoder([query], "query")[0], encoder(texts, "passage"))
    citations = cands.citations.to_numpy(dtype=float)
    return {
        "openalex": ids,
        "citas": rank_ids(ids, citations),
        "bm25": rank_ids(ids, bm25),
        "denso": rank_ids(ids, dense),
        "combinado": rank_ids(ids, start_here_score(dense, p_intro, citations)),
    }


def topic_metrics(cands: pd.DataFrame, gold: frozenset[str], orders: dict, k: int = 20) -> dict:
    ids = set(cands.id)
    return {
        "ceiling": len(ids & gold) / len(gold),
        "abstract_frac": float((cands.abstract != "").mean()) if len(cands) else 0.0,
        "n_candidates": len(cands),
        "orders": {
            name: {"recall": recall_at_k(r, set(gold), k), "ndcg": ndcg_at_k(r, set(gold), k)}
            for name, r in orders.items()
        },
    }
```

- [ ] **Step 5: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_readings.py -q`
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): lecturas por tema de OpenAlex, candidatos sin futuro y orden «empieza aquí»"
```

---

### Task 10: Ruta de conceptos y CLI `ruta`

**Files:**
- Create: `M/src/research_paths/concepts.py`, `M/src/research_paths/cli.py`,
  `M/src/research_paths/__main__.py`
- Modify: `M/tests/conftest.py` (agregar `FakeWiki` y la fixture `wiki_world`)
- Create: `M/tests/test_concepts.py`, `M/tests/test_cli.py`

**Interfaces:**
- Consumes: `Wiki` (o `FakeWiki`), `Encoder`, `cosine_scores`, `predict_proba`, `tiny_model`,
  `load_classifier`, `default_device`, `OpenAlex`, `works_frame`, `start_here_score`,
  `require_env`, `MissingKeyError`.
- Produces: `concept_text(title, extract) -> str`; `resolve_topic(query, wiki) -> (str | None,
  list[str])`; `candidate_concepts(topic, wiki, encoder, k=8, pool=30) -> list[str]`;
  `prereq_matrix(concepts, texts, model, tokenizer, *, max_length=256, device="cpu") -> np.ndarray`
  (`P[i, j]` = P(concepts[j] es prerrequisito de concepts[i])); `order_route(concepts, P) ->
  list[str]` (**versión mínima + TODO del usuario**); `prerequisite_of(concepts, P, threshold=0.5)
  -> dict[str, list[str]]`; `find_readings(concept_es, concept_en, oa, encoder, intro, *, k=3,
  n=25, device="cpu") -> pd.DataFrame`; `cli.main(argv=None, *, wiki=None, oa=None, encoder=None,
  prereq=None, intro=None, models_dir=None, device=None, out=None) -> int` (0 ok, 1 tema no
  encontrado, 2 falta modelo o clave).

- [ ] **Step 1: Wikipedia falsa en `conftest.py`**

Agregar al final de `M/tests/conftest.py`:

```python
class FakeWiki:
    """Wikipedia en memoria: `pages[título] = {"extract", "links", "langlinks": {lang: título}}`."""

    def __init__(self, pages: dict[str, dict], disambiguation=()):
        self.pages, self.disambiguation = pages, set(disambiguation)

    def resolve(self, titles):
        return {t: t if t in self.pages and t not in self.disambiguation else None for t in titles}

    def extracts(self, titles):
        return {t: self.pages.get(t, {}).get("extract", "") for t in titles}

    def links(self, title):
        return list(self.pages.get(title, {}).get("links", []))

    def langlinks(self, titles, target):
        return {t: self.pages.get(t, {}).get("langlinks", {}).get(target) for t in titles}

    def search(self, query, limit=5):
        return [t for t in self.pages if query.lower() in t.lower()][:limit]


def build_wiki_world() -> FakeWiki:
    pages = {
        "Aprendizaje por refuerzo": {
            "extract": "El aprendizaje por refuerzo estudia agentes que aprenden por recompensa.",
            "links": ["Proceso de decisión de Markov", "Probabilidad", "Programación dinámica",
                      "Estados Unidos", "Recompensa"],
        },
        "Proceso de decisión de Markov": {
            "extract": "Un proceso de decisión de Markov modela decisiones con probabilidad.",
            "links": ["Probabilidad", "Aprendizaje por refuerzo"],
            "langlinks": {"en": "Markov decision process"},
        },
        "Probabilidad": {"extract": "La probabilidad mide la incertidumbre.", "links": []},
        "Programación dinámica": {
            "extract": "La programación dinámica resuelve problemas por subproblemas.",
            "links": ["Aprendizaje por refuerzo"],
        },
        "Estados Unidos": {"extract": "", "links": []},
        "Recompensa": {"extract": "La recompensa es la señal que guía al agente.", "links": []},
        "Mercurio": {"extract": "Mercurio puede referirse a…", "links": ["Mercurio (planeta)"]},
        "Mercurio (planeta)": {"extract": "Planeta.", "links": []},
    }  # fmt: skip
    return FakeWiki(pages, disambiguation={"Mercurio"})


@pytest.fixture
def wiki_world():
    return build_wiki_world()
```

- [ ] **Step 2: Tests de `concepts.py`**

`M/tests/test_concepts.py`:

```python
"""Tema → conceptos candidatos → matriz de prerrequisitos → ruta."""

import numpy as np
import pytest

from research_paths.concepts import (
    candidate_concepts,
    concept_text,
    order_route,
    prereq_matrix,
    prerequisite_of,
    resolve_topic,
)
from research_paths.retrieval import hashing_encoder


def test_concept_text():
    assert concept_text("Null_hypothesis", "La hipótesis nula…") == "Null hypothesis. La hipótesis nula…"
    assert concept_text("Media", "") == "Media"


def test_resolve_topic_suggests_on_disambiguation(wiki_world):
    assert resolve_topic("Aprendizaje por refuerzo", wiki_world) == ("Aprendizaje por refuerzo", [])
    title, suggestions = resolve_topic("Mercurio", wiki_world)
    assert title is None and "Mercurio (planeta)" in suggestions


def test_candidate_concepts_skip_pages_without_text_and_prefer_related(wiki_world):
    out = candidate_concepts("Aprendizaje por refuerzo", wiki_world, hashing_encoder(), k=3)
    assert "Estados Unidos" not in out  # sin extracto
    assert len(out) == 3 and "Aprendizaje por refuerzo" not in out


def test_order_route_puts_a_chain_in_order():
    concepts = ["C", "A", "B"]  # A → B → C (A es prerrequisito de B, B de C)
    P = np.zeros((3, 3))
    P[0, 2] = P[0, 1] = 0.9  # C necesita B y A
    P[2, 1] = 0.9  # B necesita A
    assert order_route(concepts, P) == ["A", "B", "C"]


def test_order_route_returns_a_permutation_even_with_cycles():
    rng = np.random.default_rng(0)
    concepts = [f"c{i}" for i in range(6)]
    P = rng.random((6, 6))
    assert sorted(order_route(concepts, P)) == sorted(concepts)


def test_prerequisite_of():
    P = np.array([[0, 0.9], [0.1, 0]])
    assert prerequisite_of(["A", "B"], P) == {"A": [], "B": ["A"]}


def test_prereq_matrix_shape_and_zero_diagonal():
    pytest.importorskip("transformers")
    from research_paths.textclf import tiny_model

    concepts = ["Probabilidad", "Recompensa", "Markov"]
    texts = {c: f"{c} texto" for c in concepts}
    model, tok = tiny_model(list(texts.values()))
    P = prereq_matrix(concepts, texts, model, tok, max_length=64)
    assert P.shape == (3, 3) and np.all(np.diag(P) == 0) and np.all((P >= 0) & (P <= 1))
```

- [ ] **Step 3: Tests de la CLI**

`M/tests/test_cli.py`:

```python
"""CLI `ruta` con mundo falso: ruta completa, desambiguación, modelos o claves faltantes."""

import io

import pytest

pytest.importorskip("transformers")

from research_paths.cli import main  # noqa: E402
from research_paths.retrieval import hashing_encoder  # noqa: E402
from research_paths.textclf import tiny_model  # noqa: E402


@pytest.fixture
def models():
    texts = ["aprendizaje por refuerzo", "probabilidad", "revisión introducción", "método"]
    return tiny_model(texts), tiny_model(texts)


def _run(argv, **kw):
    out = io.StringIO()
    code = main(argv, out=out, device="cpu", **kw)
    return code, out.getvalue()


def test_route_with_readings(wiki_world, oa_world, models):
    prereq, intro = models
    # Las lecturas de OpenAlex falso se buscan por texto; el falso ignora `search` y filtra por tipo.
    code, text = _run(
        ["ruta", "Aprendizaje por refuerzo", "--conceptos", "3", "--lecturas", "2"],
        wiki=wiki_world, oa=oa_world, encoder=hashing_encoder(), prereq=prereq, intro=intro,
    )
    assert code == 0
    assert "Ruta para «Aprendizaje por refuerzo»" in text
    assert text.count("Lecturas:") == 3
    assert "citas" in text


def test_route_without_readings_needs_no_key(wiki_world, models, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    prereq, intro = models
    code, text = _run(
        ["ruta", "Aprendizaje por refuerzo", "--sin-lecturas"],
        wiki=wiki_world, encoder=hashing_encoder(), prereq=prereq, intro=intro,
    )
    assert code == 0 and "Lecturas:" not in text


def test_disambiguation_suggests_titles(wiki_world, models):
    prereq, intro = models
    code, text = _run(
        ["ruta", "Mercurio", "--sin-lecturas"],
        wiki=wiki_world, encoder=hashing_encoder(), prereq=prereq, intro=intro,
    )
    assert code == 1 and "¿Quisiste decir?" in text and "Mercurio (planeta)" in text


def test_missing_models_explain_what_to_run(wiki_world, tmp_path):
    code, text = _run(["ruta", "Aprendizaje por refuerzo"], wiki=wiki_world, models_dir=tmp_path)
    assert code == 2 and "run.py" in text


def test_missing_openalex_key_is_explained(wiki_world, models, monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    prereq, intro = models
    code, text = _run(
        ["ruta", "Aprendizaje por refuerzo"],
        wiki=wiki_world, encoder=hashing_encoder(), prereq=prereq, intro=intro,
    )
    assert code == 2 and "OPENALEX_API_KEY" in text
```

- [ ] **Step 4: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_concepts.py 02-semestre-2/02-pln/tests/test_cli.py -q`
Expected: ERROR `ModuleNotFoundError` (`research_paths.concepts`, `research_paths.cli`).

- [ ] **Step 5: Implementar `concepts.py`**

```python
"""Ruta de conceptos: candidatos desde la página del tema y orden por prerrequisitos."""

import numpy as np

from research_paths.retrieval import Encoder, cosine_scores

MUTUAL_BONUS = 0.1  # un concepto que enlaza de vuelta al tema sube un poco


def concept_text(title: str, extract: str) -> str:
    name = title.replace("_", " ")
    return f"{name}. {extract}" if extract else name


def resolve_topic(query: str, wiki) -> tuple[str | None, list[str]]:
    """Título canónico del tema, o None y hasta 5 sugerencias (no existe o es desambiguación)."""
    title = wiki.resolve([query])[query]
    return (title, []) if title else (None, wiki.search(query, 5))


def candidate_concepts(topic: str, wiki, encoder: Encoder, k: int = 8, pool: int = 30) -> list[str]:
    """Conceptos enlazados desde el tema, por similitud con él (+ bono si enlazan de vuelta)."""
    links = [x for x in wiki.links(topic) if x != topic]
    if not links:
        return []
    texts = wiki.extracts([topic, *links])
    have = [x for x in links if texts.get(x)]
    if not have:
        return []
    query = encoder([concept_text(topic, texts[topic])], "query")[0]
    sims = cosine_scores(query, encoder([concept_text(x, texts[x]) for x in have], "passage"))
    scored = []
    for i in np.argsort(-sims, kind="mergesort")[:pool]:
        mutual = topic in wiki.links(have[i])
        scored.append((float(sims[i]) + MUTUAL_BONUS * mutual, have[i]))
    scored.sort(key=lambda s: -s[0])
    return [name for _, name in scored[:k]]


def prereq_matrix(
    concepts: list[str], texts: dict[str, str], model, tokenizer, *, max_length: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """P[i, j] = P(concepts[j] es prerrequisito de concepts[i]); diagonal 0."""
    from research_paths.textclf import predict_proba

    n = len(concepts)
    P = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if pairs:
        a = [concept_text(concepts[i], texts.get(concepts[i], "")) for i, _ in pairs]
        b = [concept_text(concepts[j], texts.get(concepts[j], "")) for _, j in pairs]
        probs = predict_proba(model, tokenizer, a, b, max_length=max_length, device=device)
        for (i, j), p in zip(pairs, probs, strict=True):
            P[i, j] = p
    return P


def order_route(concepts: list[str], prereq_proba: np.ndarray) -> list[str]:
    """Orden de lectura: primero lo que es prerrequisito de más conceptos.

    `prereq_proba[i, j]` = P(concepts[j] es prerrequisito de concepts[i]).

    TODO(usuario): esta es la versión mínima (contar «victorias» por encima de 0,5). Las
    probabilidades por pares pueden ser inconsistentes (A→B, B→C y C→A). Alternativas: orden
    topológico tras romper cada ciclo por su arista más débil, Borda ponderado por probabilidad,
    o un ranking tipo Bradley-Terry. Debe devolver una permutación de `concepts`.
    """
    P = np.array(prereq_proba, dtype=float)
    np.fill_diagonal(P, 0)
    wins = (P > 0.5).sum(axis=0)
    order = sorted(range(len(concepts)), key=lambda j: (-wins[j], j))
    return [concepts[j] for j in order]


def prerequisite_of(concepts: list[str], P: np.ndarray, threshold: float = 0.5) -> dict[str, list[str]]:
    """Para cada concepto, los conceptos de los que es prerrequisito."""
    return {
        c: [concepts[i] for i in range(len(concepts)) if i != j and P[i, j] > threshold]
        for j, c in enumerate(concepts)
    }
```

Verificación a mano del test de la cadena: `wins` = columnas con P > 0,5 → A (col 1) = 2,
B (col 2) = 1, C (col 0) = 0 → [A, B, C]. ✓

- [ ] **Step 6: Implementar `cli.py` y `__main__.py`**

`M/src/research_paths/cli.py`:

```python
"""CLI: `uv run python -m research_paths ruta "tema"` → conceptos en orden y lecturas."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from research_paths.concepts import (
    candidate_concepts,
    order_route,
    prereq_matrix,
    prerequisite_of,
    resolve_topic,
)
from research_paths.http import MissingKeyError, require_env
from research_paths.openalex import KEY_URL
from research_paths.readings import start_here_score, works_frame
from research_paths.retrieval import cosine_scores

MODULE = "02-semestre-2/02-pln"
KINDS = {"review": "revisión", "book": "libro", "article": "artículo"}


def find_readings(
    concept_es: str, concept_en: str | None, oa, encoder, intro, *, k: int = 3, n: int = 25,
    device: str = "cpu",
) -> pd.DataFrame:
    """Candidatos de OpenAlex (búsqueda en es y en) ordenados por `start_here_score`."""
    from research_paths.textclf import predict_proba

    works = []
    for query in dict.fromkeys(q for q in (concept_es, concept_en) if q):
        works += oa.works(filter="type:article|review|book", search=query, max_results=n)
    cands = works_frame(works).drop_duplicates("id")
    cands = cands[cands.title != ""].reset_index(drop=True)
    if cands.empty:
        return cands
    texts = cands.text.tolist()
    p = predict_proba(*intro, texts, device=device)
    rel = cosine_scores(encoder([concept_es], "query")[0], encoder(texts, "passage"))
    cands = cands.assign(p_intro=p, score=start_here_score(rel, p, cands.citations.to_numpy()))
    return cands.sort_values("score", ascending=False, kind="mergesort").head(k)


def _reason(row) -> str:
    kind = KINDS.get(row.type, row.type or "trabajo")
    p = f"{row.p_intro:.2f}".replace(".", ",")
    return f"{kind}, {row.citations} citas, P(introductorio) {p}"


def _load_models(models_dir: Path):
    from research_paths.textclf import load_classifier

    paths = [models_dir / "prereq-es", models_dir / "intro"]
    if not all(p.is_dir() for p in paths):
        return None
    return tuple(load_classifier(p) for p in paths)


def main(
    argv=None, *, wiki=None, oa=None, encoder=None, prereq=None, intro=None, models_dir=None,
    device=None, out=None,
) -> int:
    out = out or sys.stdout
    parser = argparse.ArgumentParser(prog="research_paths", description="Por dónde empezar un tema.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ruta = sub.add_parser("ruta", help="ruta de conceptos y lecturas para un tema")
    ruta.add_argument("tema", help="título de Wikipedia en español, p. ej. «Aprendizaje por refuerzo»")
    ruta.add_argument("--conceptos", type=int, default=8)
    ruta.add_argument("--lecturas", type=int, default=3)
    ruta.add_argument("--sin-lecturas", action="store_true", help="no consultar OpenAlex")
    args = parser.parse_args(argv)

    from common import data_dir

    base = data_dir(MODULE)
    if prereq is None or intro is None:
        models_dir = Path(models_dir) if models_dir else base / "modelos"
        loaded = _load_models(models_dir)
        if loaded is None:
            print(
                f"No hay modelos entrenados en {models_dir}. Corre primero:\n"
                f"  uv run python {MODULE}/proyecto/run.py",
                file=out,
            )
            return 2
        prereq, intro = loaded
    if device is None:
        from research_paths.textclf import default_device

        device = default_device()
    try:
        if oa is None and not args.sin_lecturas:
            from research_paths.http import CachedClient
            from research_paths.openalex import OpenAlex

            key = require_env("OPENALEX_API_KEY", KEY_URL)
            oa = OpenAlex(CachedClient(base / "cache" / "openalex"), key)
    except MissingKeyError as err:
        print(err, file=out)
        return 2
    if wiki is None:
        from research_paths.http import CachedClient
        from research_paths.wiki import Wiki

        wiki = Wiki(CachedClient(base / "cache" / "wiki-es"), "es")
    if encoder is None:
        from research_paths.retrieval import e5_encoder

        encoder = e5_encoder(device=device)
    return _route(args, wiki, oa, encoder, prereq, intro, device, out)


def _route(args, wiki, oa, encoder, prereq, intro, device, out) -> int:
    title, suggestions = resolve_topic(args.tema, wiki)
    if title is None:
        print(f"No encontré «{args.tema}» en Wikipedia en español (o es una desambiguación).", file=out)
        if suggestions:
            print("¿Quisiste decir?: " + "; ".join(suggestions), file=out)
        return 1
    concepts = candidate_concepts(title, wiki, encoder, k=args.conceptos)
    if not concepts:
        print(f"«{title}» no enlaza conceptos con texto: no hay ruta.", file=out)
        return 1
    texts = wiki.extracts(concepts)
    P = prereq_matrix(concepts, texts, *prereq, device=device)
    route = order_route(concepts, P)
    before = prerequisite_of(concepts, P)
    english = {} if args.sin_lecturas else wiki.langlinks(route, "en")
    print(f"Ruta para «{title}» ({len(route)} conceptos)\n", file=out)
    for i, concept in enumerate(route, 1):
        print(f"{i}. {concept}", file=out)
        if before[concept]:
            print(f"   Prerrequisito de: {', '.join(before[concept])}", file=out)
        if args.sin_lecturas:
            continue
        found = find_readings(
            concept, english.get(concept), oa, encoder, intro, k=args.lecturas, device=device
        )
        print("   Lecturas:", file=out)
        if found.empty:
            print("   - sin lecturas encontradas", file=out)
        for row in found.itertuples():
            year = row.year or "s. f."
            print(f"   - «{row.title}» ({year}) — {_reason(row)}", file=out)
        print(file=out)
    return 0
```

`M/src/research_paths/__main__.py`:

```python
"""Permite `python -m research_paths ...`."""

from research_paths.cli import main

raise SystemExit(main())
```

- [ ] **Step 7: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_concepts.py 02-semestre-2/02-pln/tests/test_cli.py -q`
Expected: `12 passed`.

- [ ] **Step 8: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): ruta de conceptos (order_route mínima, TODO del usuario) y CLI ruta"
```

---

### Task 11: Experimentos (`experiments.py`), `run.py` y smoke

**Files:**
- Create: `M/src/research_paths/experiments.py`
- Create: `M/proyecto/run.py`
- Modify: `M/tests/conftest.py` (agregar `build_alcpl_world` y la fixture `world`)
- Create: `M/tests/test_experiments.py`, `M/tests/test_run.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: `Context(...)` (ver código); `load_concepts(ctx) -> dict` (claves `pairs, stats,
  text_es, text_en, links_es`); `run_layer1(ctx, data) -> (pd.DataFrame, dict)`;
  `train_final_prereq(ctx, data) -> None` (guarda `models_dir/prereq-es`);
  `run_layer2(ctx) -> (pd.DataFrame, dict, pd.DataFrame)`; `hypotheses(preds, topics_df, ctx) ->
  dict`; `intro_agreement(sample, labels) -> dict`; `run_all(ctx) -> dict` (escribe
  `out_dir/metrics.json` con `allow_nan=False`, `capa1_predicciones.csv`, `capa2_temas.csv`,
  `intro_muestra.csv`, `f1_por_dominio.png`, `recall_por_orden.png`).

- [ ] **Step 1: Mundo AL-CPL sintético en `conftest.py`**

Agregar arriba de `M/tests/conftest.py` los imports `from pathlib import Path` y
`from types import SimpleNamespace`, y al final:

```python
def build_alcpl_world(raw: Path) -> tuple[FakeWiki, FakeWiki]:
    """4 dominios × 5 conceptos en cadena (nivel 0 → 4). Escribe `raw/<dominio>.pairs/.preqs` y
    devuelve (wiki_en, wiki_es). Un concepto no tiene página ES (se descarta)."""
    from research_paths.alcpl import DOMAINS

    raw.mkdir(parents=True, exist_ok=True)
    en_pages, es_pages = {}, {}
    for d, domain in enumerate(DOMAINS):
        names = [f"{domain}_c{k}" for k in range(5)]
        pairs, preqs = [], []
        for i in range(5):
            for j in range(5):
                if i > j:  # j es prerrequisito de i (nivel menor)
                    pairs += [(names[i], names[j]), (names[j], names[i])]
                    preqs.append((names[i], names[j]))
        (raw / f"{domain}.pairs").write_text("".join(f"{a},{b}\n" for a, b in pairs))
        (raw / f"{domain}.preqs").write_text("".join(f"{a},{b}\n" for a, b in preqs))
        for k, name in enumerate(names):
            es = f"Concepto {d}-{k}"
            missing_es = d == 0 and k == 4
            en_pages[name] = {
                "extract": f"Concept of level {k} in {domain}.",
                "langlinks": {} if missing_es else {"es": es},
            }
            if not missing_es:
                es_pages[es] = {
                    "extract": f"Concepto de nivel {k} " + "básico " * (5 - k),
                    "links": [f"Concepto {d}-{m}" for m in range(k)],
                }
    return FakeWiki(en_pages), FakeWiki(es_pages)


class StubJudge:
    """Juez LLM falso: dice «sí» cuando el texto de B parece más básico que el de A."""

    def judge(self, items):
        return [ta.count("básico") < tb.count("básico") for _, ta, _, tb in items]


@pytest.fixture
def worlds():
    """Constructores de mundos falsos (los tests no pueden importar `conftest` por nombre)."""
    return SimpleNamespace(alcpl=build_alcpl_world, oa=build_oa_world, judge=StubJudge)
```

- [ ] **Step 2: Tests**

`M/tests/test_experiments.py`:

```python
"""Smoke de los experimentos completos sobre un mundo sintético (sin red, modelos diminutos)."""

import json

import pytest

pytest.importorskip("transformers")

from research_paths.experiments import Context, intro_agreement, run_all  # noqa: E402
from research_paths.retrieval import hashing_encoder  # noqa: E402
from research_paths.textclf import TrainConfig, tiny_model  # noqa: E402


@pytest.fixture
def ctx(tmp_path, worlds):
    wiki_en, wiki_es = worlds.alcpl(tmp_path / "data" / "alcpl")
    return Context(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "resultados",
        models_dir=tmp_path / "data" / "modelos",
        wiki_en=wiki_en,
        wiki_es=wiki_es,
        oa=worlds.oa(),
        encoder=hashing_encoder(),
        make_model=lambda texts: tiny_model(texts),
        train_cfg=TrainConfig(model_name="tiny", max_length=64, epochs=1, batch_size=8, lr=3e-3),
        judge=worlds.judge(),
        n_boot=50,
        per_field=2,
        intro_topics_per_field=2,
        fields=("17", "26"),
        log=lambda *_: None,
    )


def test_run_all_writes_complete_metrics_without_nan(ctx):
    metrics = run_all(ctx)
    on_disk = json.loads((ctx.out_dir / "metrics.json").read_text())
    assert on_disk == json.loads(json.dumps(metrics))
    cov = metrics["capa1"]["cobertura"]
    assert cov["concepts_unmapped"] == 1 and cov["pairs_kept"] < cov["pairs"]
    for model in ("prevalencia", "refd", "embeddings", "cross_es", "cross_en", "llm"):
        assert set(metrics["capa1"]["modelos"][model]) >= {"f1", "ap", "por_dominio"}
    assert metrics["capa2"]["n_temas"] == 4
    assert set(metrics["capa2"]["ordenes"]) == {"openalex", "citas", "bm25", "denso", "combinado"}
    for h in ("H1", "H2", "H3", "H4"):
        assert {"delta", "ci", "p", "p_holm", "se_apoya"} <= set(metrics["hipotesis"][h])
    for name in ("capa1_predicciones.csv", "capa2_temas.csv", "intro_muestra.csv",
                 "f1_por_dominio.png", "recall_por_orden.png"):  # fmt: skip
        assert (ctx.out_dir / name).is_file()
    assert (ctx.models_dir / "prereq-es").is_dir() and (ctx.models_dir / "intro").is_dir()


def test_without_llm_h2_is_not_evaluated(ctx):
    ctx.judge = None
    metrics = run_all(ctx)
    assert metrics["hipotesis"]["H2"] == {
        "descripcion": metrics["hipotesis"]["H2"]["descripcion"],
        "evaluada": False,
    }
    assert "llm" not in metrics["capa1"]["modelos"]


def test_intro_agreement():
    import pandas as pd

    sample = pd.DataFrame({"id": ["a", "b", "c", "d"], "p_intro": [0.9, 0.8, 0.2, 0.1]})
    labels = pd.DataFrame({"id": ["a", "b", "c", "d"], "sirve": [1, 0, 0, 0]})
    out = intro_agreement(sample, labels)
    assert out["n"] == 4 and out["positivos"] == 1 and out["ap"] == pytest.approx(1.0)
```

`M/tests/test_run.py`:

```python
"""`run.py`: ayuda y error claro si falta una clave (antes de gastar tiempo)."""

import importlib.util
from pathlib import Path

import pytest

RUN = Path(__file__).parents[1] / "proyecto" / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("pln_run", RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as err:
        _load().main(["--help"])
    assert err.value.code == 0


def test_missing_keys_fail_fast_with_instructions(monkeypatch, capsys):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _load().main([]) == 2
    assert "OPENALEX_API_KEY" in capsys.readouterr().out


def test_without_llm_only_openalex_key_is_required(monkeypatch, capsys):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _load().main(["--sin-llm"]) == 2
    assert "ANTHROPIC_API_KEY" not in capsys.readouterr().out
```

- [ ] **Step 3: Verificar que fallan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_experiments.py 02-semestre-2/02-pln/tests/test_run.py -q`
Expected: ERROR `ModuleNotFoundError: No module named 'research_paths.experiments'` y
`FileNotFoundError` por `proyecto/run.py`.

- [ ] **Step 4: Implementar `experiments.py`**

`M/src/research_paths/experiments.py`:

```python
"""Experimentos del proyecto: capa 1 (AL-CPL), capa 2 (OpenAlex) e hipótesis → resultados/.

Todo entra por `Context`, así el smoke corre con Wikipedia/OpenAlex falsos, un codificador de
hashing y modelos diminutos, y `proyecto/run.py` con los reales.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import save_fig  # noqa: E402
from research_paths.alcpl import DOMAINS, folds, load_alcpl, map_concepts, split_fold  # noqa: E402
from research_paths.concepts import concept_text  # noqa: E402
from research_paths.evaluation import (  # noqa: E402
    average_precision,
    best_threshold,
    cluster_bootstrap,
    f1_binary,
    holm,
)
from research_paths.prereq.features import fit_pair_logreg, pair_features, select_C  # noqa: E402
from research_paths.prereq.refd import refd_scores  # noqa: E402
from research_paths.readings import (  # noqa: E402
    EVAL_FIELDS,
    ORDERS,
    intro_training_set,
    ranking_orders,
    sample_eval_topics,
    topic_candidates,
    topic_metrics,
)
from research_paths.retrieval import Encoder  # noqa: E402
from research_paths.textclf import (  # noqa: E402
    TrainConfig,
    predict_proba,
    save_classifier,
    train_classifier,
)


@dataclass
class Context:
    data_dir: Path
    out_dir: Path
    models_dir: Path
    wiki_en: object
    wiki_es: object
    oa: object
    encoder: Encoder
    make_model: Callable[[list[str]], tuple]  # textos de entrenamiento → (modelo, tokenizador)
    train_cfg: TrainConfig
    judge: object | None = None
    device: str = "cpu"
    seed: int = 42
    n_boot: int = 2000
    per_field: int = 15
    intro_topics_per_field: int = 25
    fields: tuple[str, ...] = EVAL_FIELDS
    log: Callable[..., None] = field(default=print)


# ---------------------------------------------------------------- capa 1 · prerrequisitos


def load_concepts(ctx: Context) -> dict:
    df = load_alcpl(ctx.data_dir / "alcpl")
    mapping = ctx.wiki_en.langlinks(sorted(set(df.a) | set(df.b)), "es")
    pairs, stats = map_concepts(df, mapping)
    es = sorted(set(pairs.a_es) | set(pairs.b_es))
    en = sorted(set(pairs.a) | set(pairs.b))
    ext_es, ext_en = ctx.wiki_es.extracts(es), ctx.wiki_en.extracts(en)
    stats |= {
        "concepts_es": len(es),
        "es_without_extract": sum(not ext_es[c] for c in es),
        "en_without_extract": sum(not ext_en[c] for c in en),
    }
    ctx.log(f"AL-CPL → ES: {stats}")
    return {
        "pairs": pairs,
        "stats": stats,
        "text_es": {c: concept_text(c, ext_es[c]) for c in es},
        "text_en": {c: concept_text(c, ext_en[c]) for c in en},
        "links_es": {c: set(ctx.wiki_es.links(c)) for c in es},
    }


def _texts(df: pd.DataFrame, texts: dict, col_a: str, col_b: str) -> tuple[list[str], list[str]]:
    return [texts[x] for x in df[col_a]], [texts[x] for x in df[col_b]]


def _cross_encoder(ctx, train, val, test, texts, col_a, col_b) -> tuple[np.ndarray, np.ndarray]:
    tr_a, tr_b = _texts(train, texts, col_a, col_b)
    va_a, va_b = _texts(val, texts, col_a, col_b)
    te_a, te_b = _texts(test, texts, col_a, col_b)
    model, tok = ctx.make_model(tr_a + tr_b)
    model, tok, _ = train_classifier(
        tr_a, train.label.to_numpy(), va_a, val.label.to_numpy(), ctx.train_cfg,
        texts_b=tr_b, val_b=va_b, model=model, tokenizer=tok, device=ctx.device, log=ctx.log,
    )
    kw = {"max_length": ctx.train_cfg.max_length, "device": ctx.device}
    return predict_proba(model, tok, va_a, va_b, **kw), predict_proba(model, tok, te_a, te_b, **kw)


def run_layer1(ctx: Context, data: dict) -> tuple[pd.DataFrame, dict]:
    pairs, text_es, text_en = data["pairs"], data["text_es"], data["text_en"]
    concepts = set(text_es)
    names = sorted(text_es)
    emb = dict(zip(names, ctx.encoder([text_es[c] for c in names], "query"), strict=True))

    def X(d):
        return pair_features(np.stack([emb[a] for a in d.a_es]), np.stack([emb[b] for b in d.b_es]))

    rows = []
    for fold in folds():
        train, val, test = split_fold(pairs, fold)
        ctx.log(f"Ronda test={fold.test}: {len(train)}/{len(val)}/{len(test)} pares")
        out = test[["domain", "a", "b", "a_es", "b_es", "label"]].copy()
        y_val = val.label.to_numpy()

        def put(name, s_val, s_test, out=out, y_val=y_val):
            out[f"score_{name}"] = s_test
            out[f"pred_{name}"] = s_test >= best_threshold(y_val, s_val)

        prev = float(train.label.mean())
        put("prevalencia", np.full(len(val), prev), np.full(len(test), prev))
        links = data["links_es"]
        put("refd", refd_scores(val, links, concepts), refd_scores(test, links, concepts))
        C = select_C(X(train), train.label, X(val), y_val)
        lr = fit_pair_logreg(X(train), train.label, C)
        put("embeddings", lr.predict_proba(X(val))[:, 1], lr.predict_proba(X(test))[:, 1])
        put("cross_es", *_cross_encoder(ctx, train, val, test, text_es, "a_es", "b_es"))
        put("cross_en", *_cross_encoder(ctx, train, val, test, text_en, "a", "b"))
        if ctx.judge is not None:
            items = [(a, text_es[a], b, text_es[b]) for a, b in zip(test.a_es, test.b_es, strict=True)]
            answers = ctx.judge.judge(items)
            out["pred_llm"] = [a is True for a in answers]
            out["score_llm"] = out.pred_llm.astype(float)
            out["llm_unparsed"] = [a is None for a in answers]
        rows.append(out)
    preds = pd.concat(rows, ignore_index=True)
    models = ["prevalencia", "refd", "embeddings", "cross_es", "cross_en"]
    if "pred_llm" in preds:
        models.append("llm")
    metrics = {"modelos": {m: _layer1_metrics(preds, m) for m in models}}
    if "llm_unparsed" in preds:
        metrics["llm_no_interpretables"] = int(preds.llm_unparsed.sum())
    return preds, metrics


def _layer1_metrics(preds: pd.DataFrame, model: str) -> dict:
    def one(d):
        y = d.label.to_numpy()
        return {
            "f1": f1_binary(y, d[f"pred_{model}"]),
            "ap": average_precision(y, d[f"score_{model}"]),
            "n": len(d),
            "positivos": int(y.sum()),
        }

    return {**one(preds), "por_dominio": {d: one(g) for d, g in preds.groupby("domain")}}


def train_final_prereq(ctx: Context, data: dict) -> None:
    """Cross-encoder ES para la CLI: entrena con 3 dominios y valida en el más pequeño."""
    pairs = data["pairs"]
    val_domain = pairs.domain.value_counts().idxmin()
    val = pairs[pairs.domain == val_domain]
    val_c = set(val.a_es) | set(val.b_es)
    train = pairs[(pairs.domain != val_domain) & ~pairs.a_es.isin(val_c) & ~pairs.b_es.isin(val_c)]
    tr_a, tr_b = _texts(train, data["text_es"], "a_es", "b_es")
    va_a, va_b = _texts(val, data["text_es"], "a_es", "b_es")
    model, tok = ctx.make_model(tr_a + tr_b)
    model, tok, _ = train_classifier(
        tr_a, train.label.to_numpy(), va_a, val.label.to_numpy(), ctx.train_cfg,
        texts_b=tr_b, val_b=va_b, model=model, tokenizer=tok, device=ctx.device, log=ctx.log,
    )
    save_classifier(model, tok, ctx.models_dir / "prereq-es")


# ---------------------------------------------------------------- capa 2 · lecturas


def run_layer2(ctx: Context) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    topics = sample_eval_topics(ctx.oa, fields=ctx.fields, per_field=ctx.per_field, seed=ctx.seed)
    ctx.log(f"Temas evaluados: {len(topics)}")
    intro = intro_training_set(
        ctx.oa, topics, fields=ctx.fields, topics_per_field=ctx.intro_topics_per_field,
        seed=ctx.seed + 1,
    )
    rng = np.random.default_rng(ctx.seed)
    tids = np.array(sorted(intro.topic_id.unique()))
    rng.shuffle(tids)
    val_topics = set(tids[: max(1, len(tids) // 5)])
    tr, va = intro[~intro.topic_id.isin(val_topics)], intro[intro.topic_id.isin(val_topics)]
    model, tok = ctx.make_model(tr.text.tolist())
    model, tok, history = train_classifier(
        tr.text.tolist(), tr.label.to_numpy(), va.text.tolist(), va.label.to_numpy(),
        ctx.train_cfg, model=model, tokenizer=tok, device=ctx.device, log=ctx.log,
    )
    save_classifier(model, tok, ctx.models_dir / "intro")
    kw = {"max_length": ctx.train_cfg.max_length, "device": ctx.device}
    rows, pools = [], []
    for t in topics:
        cands = topic_candidates(ctx.oa, t)
        p = predict_proba(model, tok, cands.text.tolist(), **kw) if len(cands) else np.zeros(0)
        m = topic_metrics(cands, t.gold, ranking_orders(cands, t.name, ctx.encoder, p))
        row = {
            "topic_id": t.topic_id, "name": t.name, "field": t.field, "review_id": t.review_id,
            "review_year": t.review_year, "n_gold": len(t.gold), "n_candidates": m["n_candidates"],
            "ceiling": m["ceiling"], "abstract_frac": m["abstract_frac"],
        }  # fmt: skip
        for name, v in m["orders"].items():
            row[f"recall_{name}"], row[f"ndcg_{name}"] = v["recall"], v["ndcg"]
        rows.append(row)
        pools.append(cands.assign(p_intro=p, topic_id=t.topic_id))
    topics_df = pd.DataFrame(rows)
    pool = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    pool = pool[pool.abstract != ""] if len(pool) else pool
    sample = pool.sample(min(100, len(pool)), random_state=ctx.seed) if len(pool) else pool
    sample = sample[["id", "topic_id", "title", "abstract", "p_intro"]] if len(sample) else sample
    metrics = {
        "n_temas": len(topics_df),
        "temas_por_campo": {k: int(v) for k, v in topics_df.field.value_counts().items()},
        "techo_medio": float(topics_df.ceiling.mean()) if len(topics_df) else 0.0,
        "fraccion_con_resumen": float(topics_df.abstract_frac.mean()) if len(topics_df) else 0.0,
        "intro": {
            "n_train": len(tr),
            "n_val": len(va),
            "ap_val": max(h["val_ap"] for h in history),
            "historia": history,
        },
        "ordenes": {
            name: {
                "recall@20": float(topics_df[f"recall_{name}"].mean()),
                "ndcg@20": float(topics_df[f"ndcg_{name}"].mean()),
            }
            for name in ORDERS
        },
    }
    return topics_df, metrics, sample


def intro_agreement(sample: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Acuerdo del clasificador introductorio con las etiquetas «¿sirve para empezar?»."""
    m = sample.merge(labels[["id", "sirve"]], on="id")
    y, p = m.sirve.to_numpy(), m.p_intro.to_numpy()
    return {
        "n": len(m),
        "positivos": int(y.sum()),
        "ap": average_precision(y, p),
        "f1@0,5": f1_binary(y, p >= 0.5),
    }


# ---------------------------------------------------------------- hipótesis


def hypotheses(preds: pd.DataFrame, topics_df: pd.DataFrame, ctx: Context) -> dict:
    rng = np.random.default_rng(ctx.seed)
    y = preds.label.to_numpy()
    concepts = preds.a_es.to_numpy()

    def f1_diff(m1, m2):
        p1, p2 = preds[f"pred_{m1}"].to_numpy(), preds[f"pred_{m2}"].to_numpy()
        return lambda idx: f1_binary(y[idx], p1[idx]) - f1_binary(y[idx], p2[idx])

    H = {
        "H1": {"descripcion": "F1(cross-encoder ES) − F1(RefD)",
               **cluster_bootstrap(concepts, f1_diff("cross_es", "refd"), rng, ctx.n_boot)},
        "H2": {"descripcion": "F1(juez LLM) − F1(cross-encoder ES)", "evaluada": False},
        "H3": {"descripcion": "F1(cross-encoder EN) − F1(cross-encoder ES)",
               **cluster_bootstrap(concepts, f1_diff("cross_en", "cross_es"), rng, ctx.n_boot)},
    }  # fmt: skip
    if "pred_llm" in preds:
        H["H2"] = {
            "descripcion": H["H2"]["descripcion"],
            **cluster_bootstrap(concepts, f1_diff("llm", "cross_es"), rng, ctx.n_boot),
        }
    diff = (topics_df.recall_combinado - topics_df.recall_citas).to_numpy()
    H["H4"] = {
        "descripcion": "recall@20(combinado) − recall@20(más citados)",
        **cluster_bootstrap(topics_df.topic_id.to_numpy(), lambda i: float(diff[i].mean()), rng, ctx.n_boot),
    }
    evaluated = [k for k in ("H1", "H2", "H3", "H4") if H[k].get("evaluada", True)]
    for k, p_holm in zip(evaluated, holm([H[k]["p"] for k in evaluated]), strict=True):
        H[k].pop("_boots")
        H[k]["p_holm"] = p_holm
        H[k]["se_apoya"] = bool(p_holm < 0.05 and H[k]["delta"] > 0)
    return H


# ---------------------------------------------------------------- figuras y orquestación


def _plot_layer1(metrics: dict, path: Path) -> None:
    models = list(metrics["modelos"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(models)
    for k, m in enumerate(models):
        vals = [metrics["modelos"][m]["por_dominio"].get(d, {}).get("f1", 0.0) for d in DOMAINS]
        ax.bar(np.arange(len(DOMAINS)) + k * width, vals, width, label=m)
    ax.set_xticks(np.arange(len(DOMAINS)) + 0.4 - width / 2, DOMAINS)
    ax.set_ylabel("F1 (dominio no visto)")
    ax.set_title("Prerrequisitos: F1 por dominio de test")
    ax.legend(ncols=3, fontsize=8)
    save_fig(fig, path)
    plt.close(fig)


def _plot_layer2(metrics: dict, path: Path) -> None:
    names = list(metrics["ordenes"])
    fig, ax = plt.subplots()
    ax.bar(names, [metrics["ordenes"][n]["recall@20"] for n in names])
    ax.axhline(metrics["techo_medio"], ls="--", c="gray", label="techo (candidatos)")
    ax.set_ylabel("recall@20 medio")
    ax.set_title("Lecturas: referencias de la revisión recuperadas")
    ax.legend()
    save_fig(fig, path)
    plt.close(fig)


def _api_stats(ctx: Context) -> dict:
    out = {}
    for name in ("wiki_en", "wiki_es", "oa"):
        stats = getattr(getattr(getattr(ctx, name), "client", None), "stats", None)
        if stats is not None:
            out[name] = dict(stats)
    judge_stats = getattr(ctx.judge, "stats", None)
    if judge_stats is not None:
        out["llm"] = dict(judge_stats)
    return out


def run_all(ctx: Context) -> dict:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_concepts(ctx)
    preds, m1 = run_layer1(ctx, data)
    train_final_prereq(ctx, data)
    topics_df, m2, sample = run_layer2(ctx)
    metrics = {
        "capa1": {"cobertura": data["stats"], **m1},
        "capa2": m2,
        "hipotesis": hypotheses(preds, topics_df, ctx),
        "api": _api_stats(ctx),
    }
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    (ctx.out_dir / "metrics.json").write_text(text + "\n")
    preds.to_csv(ctx.out_dir / "capa1_predicciones.csv", index=False)
    topics_df.to_csv(ctx.out_dir / "capa2_temas.csv", index=False)
    sample.to_csv(ctx.out_dir / "intro_muestra.csv", index=False)
    _plot_layer1(m1, ctx.out_dir / "f1_por_dominio.png")
    _plot_layer2(m2, ctx.out_dir / "recall_por_orden.png")
    return json.loads(text)
```

Notas de implementación:
- Los `# noqa: E402` existen porque `matplotlib.use("Agg")` va antes de importar `pyplot`. Si
  `common.save_fig` ya importa `pyplot` sin backend fijo, mover `matplotlib.use("Agg")` no cambia
  nada en CI (sin pantalla, matplotlib elige Agg); en ese caso quitar `matplotlib.use` y los
  `noqa`, y registrar el Ruling.
- `ctx.judge.judge` recibe solo los pares de **test** de cada ronda: el LLM no se entrena, así que
  cada par se juzga una vez.

- [ ] **Step 5: Implementar `proyecto/run.py`**

`M/proyecto/run.py`:

```python
"""Experimentos del motor de rutas: capa 1 (AL-CPL), capa 2 (OpenAlex) e hipótesis → resultados/.

Requiere `data/download.py` (AL-CPL) y las claves OPENALEX_API_KEY y ANTHROPIC_API_KEY en el
entorno (con --sin-llm no se usa Anthropic y H2 queda «no evaluada"). Wikipedia, OpenAlex y el
LLM se guardan en data/cache/: la segunda ejecución no usa red ni gasta presupuesto.

Uso:
    uv run python 02-semestre-2/02-pln/proyecto/run.py [--sin-llm] [--por-campo 15]
    uv run python 02-semestre-2/02-pln/proyecto/run.py --acuerdo-intro   # tras etiquetar la muestra
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from common import data_dir
from research_paths.http import CachedClient, MissingKeyError, require_env

MODULE = "02-semestre-2/02-pln"
OUT = Path(__file__).resolve().parent / "resultados"
LABELS = Path(__file__).resolve().parent / "intro_etiquetas.csv"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sin-llm", action="store_true", help="no usar el juez LLM (H2 no evaluada)")
    parser.add_argument("--por-campo", type=int, default=15, help="temas de OpenAlex por campo")
    parser.add_argument(
        "--acuerdo-intro", action="store_true",
        help="solo agrega a metrics.json el acuerdo con proyecto/intro_etiquetas.csv",
    )  # fmt: skip
    args = parser.parse_args(argv)

    if args.acuerdo_intro:
        from research_paths.experiments import intro_agreement

        metrics = json.loads((OUT / "metrics.json").read_text())
        sample = pd.read_csv(OUT / "intro_muestra.csv")
        metrics["capa2"]["intro"]["acuerdo_manual"] = intro_agreement(sample, pd.read_csv(LABELS))
        text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
        (OUT / "metrics.json").write_text(text + "\n")
        print(metrics["capa2"]["intro"]["acuerdo_manual"])
        return 0

    try:
        oa_key = require_env("OPENALEX_API_KEY", "https://openalex.org/settings/api")
        if not args.sin_llm:
            require_env("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys")
    except MissingKeyError as err:
        print(err)
        return 2

    from research_paths.experiments import Context, run_all
    from research_paths.openalex import OpenAlex
    from research_paths.prereq.llm import LLMJudge
    from research_paths.retrieval import e5_encoder
    from research_paths.textclf import TrainConfig, default_device, load_pretrained
    from research_paths.wiki import Wiki

    base = data_dir(MODULE)
    if not (base / "alcpl").is_dir():
        print(f"Falta AL-CPL: uv run python {MODULE}/data/download.py")
        return 2
    device = default_device()
    cache = base / "cache"
    ctx = Context(
        data_dir=base,
        out_dir=OUT,
        models_dir=base / "modelos",
        wiki_en=Wiki(CachedClient(cache / "wiki-en"), "en"),
        wiki_es=Wiki(CachedClient(cache / "wiki-es"), "es"),
        oa=OpenAlex(CachedClient(cache / "openalex"), oa_key),
        encoder=e5_encoder(device=device),
        make_model=lambda texts: load_pretrained("xlm-roberta-base"),
        train_cfg=TrainConfig(),
        judge=None if args.sin_llm else LLMJudge(cache / "llm"),
        device=device,
        per_field=args.por_campo,
    )
    metrics = run_all(ctx)
    for h, v in metrics["hipotesis"].items():
        print(h, v.get("descripcion"), {k: v[k] for k in ("delta", "p_holm", "se_apoya") if k in v})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Verificar que pasan**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_experiments.py 02-semestre-2/02-pln/tests/test_run.py -q`
Expected: `6 passed` (el smoke tarda segundos: modelos diminutos de 1 época, 50 réplicas).

- [ ] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln
git commit -m "feat(pln): experimentos por capas, hipótesis con bootstrap y Holm, y run.py"
```

---
### Task 12: Notebooks (5 smoke + 1 con modelos preentrenados)

**Files:**
- Create: `M/notebooks/01-tokenizacion.ipynb`, `02-embeddings.ipynb`, `03-atencion.ipynb`,
  `04-clasificacion-afinada.ipynb`, `05-recuperacion.ipynb` (smoke) y
  `06-modelos-preentrenados.ipynb` (no smoke: descarga XLM-R y e5)
- Delete: `M/notebooks/.gitkeep`

**Interfaces:**
- Consumes: `train_bpe`, `encode`, `scaled_dot_product_attention`, `causal_mask`,
  `multi_head_attention`, `BM25`, `simple_tokenize`, `hashing_encoder`, `cosine_scores`,
  `recall_at_k`, `ndcg_at_k`, `tiny_model`, `train_classifier`, `predict_proba`, `e5_encoder`.

- [ ] **Step 1: Generador de notebooks**

Guardar en el workspace del plan (no se versiona) `build_notebooks.py` y ejecutarlo con
`uv run python <workspace>/build_notebooks.py`:

```python
"""Genera los notebooks del módulo PLN (celdas cortas, semilla 42)."""

from pathlib import Path

import nbformat as nbf

M = Path("02-semestre-2/02-pln/notebooks")
SETUP = "import numpy as np\nimport matplotlib.pyplot as plt\n\nfrom common import set_seed, setup_style\n\nsetup_style()\nrng = set_seed(42)"


def nb(name, cells, smoke=True):
    book = nbf.v4.new_notebook()
    book.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        **({"smoke": True} if smoke else {}),
    }
    book.cells = [
        nbf.v4.new_markdown_cell(c[3:]) if c.startswith("md:") else nbf.v4.new_code_cell(c)
        for c in cells
    ]
    nbf.write(book, M / name)


CORPUS = '''CORPUS = [
    "el aprendizaje por refuerzo estudia agentes que aprenden por recompensa",
    "la investigación en aprendizaje automático usa datos para aprender modelos",
    "la información de los datos permite aprender representaciones",
    "los modelos de lenguaje aprenden representaciones de palabras",
    "la moniliasis del cacao es una enfermedad de las mazorcas",
    "las enfermedades del cacao reducen la producción de las fincas",
]'''

nb("01-tokenizacion.ipynb", [
    "md:# Tokenización: BPE desde cero\n\nUn modelo de lenguaje no ve palabras sino *tokens*. BPE "
    "parte de caracteres y fusiona el par más frecuente una y otra vez. Aquí lo implementamos "
    "(`research_paths.bpe`) y verificamos que da **exactamente** las mismas fusiones que la "
    "librería `tokenizers` de Hugging Face.",
    SETUP,
    CORPUS,
    "from research_paths.bpe import encode, train_bpe\n\nmerges = train_bpe(CORPUS, 40)\nmerges[:12]",
    "md:Las primeras fusiones son terminaciones frecuentes del español (`es`, `an`, `de`…). "
    "Con más fusiones aparecen palabras completas.",
    "for w in [\"aprendizaje\", \"investigación\", \"moniliasis\", \"recompensas\"]:\n"
    "    print(f\"{w:15s} → {encode(w, merges)}\")",
    "md:## Contra `tokenizers`\n\nEl desempate importa: ante dos pares igual de frecuentes, "
    "`tokenizers` elige el de ids menores (alfabeto ordenado y luego orden de creación).",
    "import json\n\nfrom tokenizers import Tokenizer, models, pre_tokenizers, trainers\n\n"
    "tok = Tokenizer(models.BPE())\ntok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()\n"
    "alphabet = {c for line in CORPUS for c in line if c != \" \"}\n"
    "tok.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=len(alphabet) + 40, "
    "min_frequency=0, show_progress=False))\n"
    "ref = [tuple(m) for m in json.loads(tok.to_str())[\"model\"][\"merges\"]]\n"
    "assert ref == merges\nprint(\"Mismas 40 fusiones que tokenizers ✓\")",
    "md:## Fertilidad: tokens por palabra\n\nMás fusiones → menos tokens por palabra, pero un "
    "vocabulario más grande. Las palabras que no están en el corpus («moniliasis») se parten en "
    "pedazos: así un modelo nunca tiene palabras desconocidas.",
    "words = \" \".join(CORPUS).split()\nsizes = [0, 10, 20, 40, 80, 160]\n"
    "fert = [len(encode(\" \".join(words), train_bpe(CORPUS, n))) / len(words) for n in sizes]\n"
    "fig, ax = plt.subplots()\nax.plot(sizes, fert, marker=\"o\")\n"
    "ax.set(xlabel=\"fusiones\", ylabel=\"tokens por palabra\", title=\"Fertilidad del BPE\");",
    "md:El notebook 06 muestra el tokenizador real de XLM-R (SentencePiece, 250 000 tokens) "
    "con palabras en español e inglés.",
])

nb("02-embeddings.ipynb", [
    "md:# Embeddings: estáticos y contextuales\n\nUn embedding estático da **un** vector por "
    "palabra. Lo construimos desde cero (PPMI + SVD) y vemos su límite: «banco» de río y "
    "«banco» de dinero comparten vector. Un embedding contextual mezcla la palabra con su "
    "contexto (atención) y los separa.",
    SETUP,
    "SENTS = [\n    \"me senté en el banco del parque junto al río\",\n"
    "    \"el banco del río estaba cubierto de hojas\",\n"
    "    \"el banco aprobó el crédito y el préstamo\",\n"
    "    \"pagué la cuota del crédito en el banco\",\n"
    "    \"el río lleva agua al parque\",\n    \"el préstamo tiene una cuota mensual\",\n] * 5",
    "from collections import Counter\nfrom itertools import combinations\n\n"
    "vocab = sorted({w for s in SENTS for w in s.split()})\nidx = {w: i for i, w in enumerate(vocab)}\n"
    "co = np.zeros((len(vocab), len(vocab)))\nfor s in SENTS:\n    ws = s.split()\n"
    "    for i, j in combinations(range(len(ws)), 2):\n        if j - i <= 3:\n"
    "            co[idx[ws[i]], idx[ws[j]]] += 1\n            co[idx[ws[j]], idx[ws[i]]] += 1\n"
    "total = co.sum()\npmi = np.log((co * total + 1e-12) / (co.sum(1, keepdims=True) * co.sum(0, keepdims=True) + 1e-12))\n"
    "ppmi = np.maximum(pmi, 0)\nU, S, _ = np.linalg.svd(ppmi)\nE = U[:, :8] * S[:8]\nE.shape",
    "def neighbors(w, k=4):\n    v = E[idx[w]]\n"
    "    sims = E @ v / (np.linalg.norm(E, axis=1) * np.linalg.norm(v) + 1e-12)\n"
    "    return [vocab[i] for i in np.argsort(-sims)[1 : k + 1]]\n\n"
    "for w in [\"río\", \"crédito\", \"banco\"]:\n    print(w, \"→\", neighbors(w))",
    "md:«banco» queda a medio camino entre río y dinero: un solo vector para dos sentidos.\n\n"
    "## Contextual: atención sobre la oración\n\nCada palabra pasa a ser un promedio de los "
    "vectores de su oración, ponderado por similitud (una cabeza de atención sin entrenar).",
    "from research_paths.attention import scaled_dot_product_attention\n\n"
    "def contextual(sentence):\n    X = np.stack([E[idx[w]] for w in sentence.split()])\n"
    "    out, _ = scaled_dot_product_attention(X, X, X)\n"
    "    return dict(zip(sentence.split(), out, strict=True))\n\n"
    "a = contextual(\"el banco del río estaba cubierto de hojas\")[\"banco\"]\n"
    "b = contextual(\"el banco aprobó el crédito y el préstamo\")[\"banco\"]\n"
    "cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))\n"
    "print(f\"coseno entre los dos «banco» contextuales: {cos:.3f} (estático: 1.000)\")",
    "md:Un transformer real apila muchas capas de esto con proyecciones aprendidas. El notebook "
    "06 muestra el espacio multilingüe de e5: «aprendizaje por refuerzo» y «reinforcement "
    "learning» quedan cerca aunque no compartan palabras.",
])

nb("03-atencion.ipynb", [
    "md:# Atención y transformer\n\nsoftmax(QKᵀ/√d)·V en NumPy, la máscara causal y varias "
    "cabezas, verificadas contra PyTorch.",
    SETUP,
    "import torch\n\nfrom research_paths.attention import causal_mask, multi_head_attention, "
    "scaled_dot_product_attention\n\nq, k, v = (rng.normal(size=(6, 8)) for _ in range(3))\n"
    "out, w = scaled_dot_product_attention(q, k, v, causal_mask(6))\n"
    "ref = torch.nn.functional.scaled_dot_product_attention(*(torch.tensor(t) for t in (q, k, v)), is_causal=True)\n"
    "print(\"diferencia máxima con torch:\", np.abs(out - ref.numpy()).max())",
    "fig, ax = plt.subplots(figsize=(4, 4))\nax.imshow(w, cmap=\"viridis\")\n"
    "ax.set(title=\"Pesos con máscara causal\", xlabel=\"clave\", ylabel=\"consulta\");",
    "md:## ¿Por qué dividir por √d?\n\nSin escalar, el producto punto crece con la dimensión y el "
    "softmax se satura: casi todo el peso va a una sola clave y el gradiente se apaga.",
    "for d in (4, 64, 1024):\n    qd, kd = rng.normal(size=(2, 50, d))\n"
    "    _, w_raw = scaled_dot_product_attention(qd * np.sqrt(d), kd, np.eye(50))  # deshace el /√d\n"
    "    _, w_ok = scaled_dot_product_attention(qd, kd, np.eye(50))\n"
    "    print(f\"d={d:5d}: peso máximo medio sin escalar {w_raw.max(-1).mean():.2f} · escalado {w_ok.max(-1).mean():.2f}\")",
    "md:## Varias cabezas\n\nCada cabeza mira el mismo texto con proyecciones distintas.",
    "x = rng.normal(size=(5, 8))\nW = [rng.normal(size=(8, 8)) for _ in range(4)]\n"
    "out, weights = multi_head_attention(x, *W, num_heads=2)\nweights.shape  # (cabezas, n, n)",
])

nb("04-clasificacion-afinada.ipynb", [
    "md:# Clasificación afinada\n\nAjustamos un transformer (XLM-R diminuto, sin descargas) a "
    "una tarea de pares: ¿el texto B es «básico» respecto de A? Es la misma maquinaria que el "
    "cross-encoder de prerrequisitos del proyecto, en miniatura. Comparamos con TF-IDF + "
    "logística.",
    SETUP,
    "WORDS = \"gato perro casa árbol río sol luna mar\".split()\n\ndef pairs(n):\n"
    "    a = [\" \".join(rng.choice(WORDS, 4)) for _ in range(n)]\n    y = rng.integers(0, 2, n)\n"
    "    b = [(\"básico elemental \" if t else \"avanzado complejo \") + \" \".join(rng.choice(WORDS, 3)) for t in y]\n"
    "    return a, b, y\n\n(A, B, Y), (VA, VB, VY) = pairs(160), pairs(60)",
    "from research_paths.textclf import TrainConfig, predict_proba, tiny_model, train_classifier\n\n"
    "model, tok = tiny_model(A + B)\ncfg = TrainConfig(model_name=\"tiny\", max_length=64, epochs=6, batch_size=16, lr=3e-3, warmup_ratio=0.0)\n"
    "model, tok, history = train_classifier(A, Y, VA, VY, cfg, texts_b=B, val_b=VB, model=model, tokenizer=tok)",
    "fig, ax = plt.subplots()\nax.plot([h[\"epoch\"] for h in history], [h[\"val_ap\"] for h in history], marker=\"o\")\n"
    "ax.set(xlabel=\"época\", ylabel=\"AP en validación\", ylim=(0, 1.05));",
    "from sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\n\n"
    "from research_paths.evaluation import average_precision\n\n"
    "vec = TfidfVectorizer().fit([a + \" \" + b for a, b in zip(A, B, strict=True)])\n"
    "clf = LogisticRegression().fit(vec.transform([a + \" \" + b for a, b in zip(A, B, strict=True)]), Y)\n"
    "p_tfidf = clf.predict_proba(vec.transform([a + \" \" + b for a, b in zip(VA, VB, strict=True)]))[:, 1]\n"
    "p_tr = predict_proba(model, tok, VA, VB, max_length=64)\n"
    "print(f\"AP TF-IDF: {average_precision(VY, p_tfidf):.3f} · AP transformer: {average_precision(VY, p_tr):.3f}\")",
    "md:Aquí ambos aciertan: la tarea es léxica. En el proyecto la diferencia aparece cuando la "
    "relación depende del significado (prerrequisitos) y no de palabras clave.",
])

nb("05-recuperacion.ipynb", [
    "md:# Recuperación: BM25 frente a búsqueda densa\n\nBM25 cuenta palabras con saturación y "
    "normalización por largo. La búsqueda densa compara vectores. Medimos ambas con recall@k y "
    "NDCG@k sobre un mini-corpus con relevancia marcada a mano.",
    SETUP,
    "DOCS = {\n    \"d1\": \"introducción al aprendizaje por refuerzo: agentes, recompensas y políticas\",\n"
    "    \"d2\": \"q-learning profundo para juegos de atari\",\n"
    "    \"d3\": \"revisión de métodos de gradiente de política\",\n"
    "    \"d4\": \"enfermedades del cacao en colombia\",\n"
    "    \"d5\": \"procesos de decisión de markov y programación dinámica\",\n"
    "    \"d6\": \"redes neuronales convolucionales para imágenes\",\n"
    "    \"d7\": \"tutorial de aprendizaje por refuerzo para principiantes\",\n"
    "    \"d8\": \"series de tiempo y pronóstico de demanda\",\n}\n"
    "RELEVANT = {\"d1\", \"d3\", \"d5\", \"d7\"}\nQUERY = \"aprendizaje por refuerzo introducción\"",
    "from research_paths.evaluation import ndcg_at_k, recall_at_k\n"
    "from research_paths.retrieval import BM25, cosine_scores, hashing_encoder, simple_tokenize\n\n"
    "ids, texts = list(DOCS), list(DOCS.values())\n"
    "bm = BM25().fit([simple_tokenize(t) for t in texts]).scores(simple_tokenize(QUERY))\n"
    "enc = hashing_encoder(512)\ndense = cosine_scores(enc([QUERY], \"query\")[0], enc(texts, \"passage\"))\n"
    "for name, s in [(\"BM25\", bm), (\"denso (hashing)\", dense)]:\n"
    "    ranked = [ids[i] for i in np.argsort(-s, kind=\"mergesort\")]\n"
    "    print(f\"{name:16s} {ranked[:4]}  recall@4={recall_at_k(ranked, RELEVANT, 4):.2f}  NDCG@4={ndcg_at_k(ranked, RELEVANT, 4):.2f}\")",
    "md:El codificador de hashing es una bolsa de palabras: no entiende que «procesos de "
    "decisión de Markov» es parte del tema. e5 (notebook 06 y el proyecto) sí lo capta, porque "
    "aprendió de millones de pares de textos relacionados. BM25 sigue siendo una línea base "
    "difícil de batir cuando la consulta comparte palabras con los documentos.",
])

nb("06-modelos-preentrenados.ipynb", [
    "md:# Modelos preentrenados (no smoke: descarga ~2 GB)\n\nEl tokenizador de XLM-R y el "
    "espacio multilingüe de e5, los dos modelos que usa el proyecto.",
    SETUP,
    "from transformers import AutoTokenizer\n\ntok = AutoTokenizer.from_pretrained(\"xlm-roberta-base\")\n"
    "for w in [\"aprendizaje\", \"learning\", \"moniliasis\", \"investigación\", \"research\", \"prerrequisito\"]:\n"
    "    print(f\"{w:15s} → {tok.tokenize(w)}\")",
    "md:El español se parte en más pedazos que el inglés: el vocabulario de XLM-R se repartió "
    "entre 100 idiomas según cuánto texto había de cada uno.",
    "from research_paths.retrieval import e5_encoder\n\nenc = e5_encoder()\n"
    "phrases = [\"aprendizaje por refuerzo\", \"reinforcement learning\", \"proceso de decisión de Markov\",\n"
    "           \"Markov decision process\", \"moniliasis del cacao\", \"gato doméstico\"]\n"
    "V = enc(phrases, \"query\")\nS = V @ V.T\nfig, ax = plt.subplots(figsize=(6, 5))\n"
    "ax.imshow(S, vmin=0.6, vmax=1, cmap=\"viridis\")\nax.set_xticks(range(6), phrases, rotation=60, ha=\"right\")\n"
    "ax.set_yticks(range(6), phrases)\nax.set_title(\"Coseno e5 (es y en en el mismo espacio)\");",
], smoke=False)
```

```bash
git rm -q 02-semestre-2/02-pln/notebooks/.gitkeep
uv run python <workspace>/build_notebooks.py
uv run ruff format 02-semestre-2/02-pln/notebooks && uv run ruff check --fix 02-semestre-2/02-pln/notebooks
```

- [ ] **Step 2: Ejecutar los smoke y guardar salidas**

Run: `for n in 01-tokenizacion 02-embeddings 03-atencion 04-clasificacion-afinada 05-recuperacion; do uv run python scripts/run_notebooks.py --inplace 02-semestre-2/02-pln/notebooks/$n.ipynb || break; done`
Expected: cada uno termina sin error en segundos. Leer las salidas y comprobar que el texto de las
celdas markdown coincide con lo que muestran (p. ej. que el coseno de los dos «banco» sea < 1, que
la AP del notebook 04 sea alta para ambos y que en el 05 la afirmación sobre BM25 vs. hashing sea
cierta con los números impresos). **Si una afirmación no se cumple, cambiar el texto, no los datos.**

El 06 se ejecuta en la Tarea 14 (necesita descargas).

- [ ] **Step 3: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest && uv run python scripts/run_notebooks.py
git add 02-semestre-2/02-pln/notebooks
git commit -m "docs(pln): notebooks de tokenización, embeddings, atención, ajuste y recuperación"
```

---

### Task 13: Ejercicios con soluciones

**Files:**
- Create: `M/ejercicios/README.md`, `M/ejercicios/ej01_wordpiece.py` … `ej06_refd.py`
- Create: `M/ejercicios/soluciones/ej01_wordpiece.py` … `ej06_refd.py`
- Create: `M/tests/test_soluciones.py`
- Delete: `M/ejercicios/soluciones/.gitkeep`

Cada stub tiene la firma con `raise NotImplementedError` y el **mismo** `check()` que su solución.

- [ ] **Step 1: Test de soluciones**

`M/tests/test_soluciones.py`:

```python
"""Ejecuta el `check()` de cada solución: garantiza que las soluciones publicadas son correctas."""

import importlib.util
from pathlib import Path

import pytest

SOLUTIONS = sorted((Path(__file__).parents[1] / "ejercicios" / "soluciones").glob("ej*.py"))


def test_there_are_six_solutions():
    assert len(SOLUTIONS) == 6


@pytest.mark.parametrize("path", SOLUTIONS, ids=lambda p: p.stem)
def test_solution_passes_its_check(path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(f"sol_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check()
```

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_soluciones.py -q`
Expected: FAIL `assert 0 == 6`.

- [ ] **Step 2: Soluciones** (el stub = mismo archivo con el cuerpo de la función reemplazado por
`raise NotImplementedError` y el docstring del ejercicio)

`soluciones/ej01_wordpiece.py`:

```python
"""Ejercicio 1: WordPiece greedy (el más largo primero, continuaciones con ##). Ver README.md."""


def wordpiece(word: str, vocab: set[str], unk: str = "[UNK]", max_chars: int = 100) -> list[str]:
    """Parte `word` en piezas del vocabulario; si algún tramo no tiene pieza, devuelve [unk]."""
    if len(word) > max_chars:
        return [unk]
    pieces, start = [], 0
    while start < len(word):
        end, piece = len(word), None
        while start < end:
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in vocab:
                piece = sub
                break
            end -= 1
        if piece is None:
            return [unk]
        pieces.append(piece)
        start = end
    return pieces


def check() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = ["[UNK]", "un", "##afable", "##a", "##fable", "casa", "##s", "ca", "##sa", "in", "##vest",
             "##iga", "##ción", "##i", "##g", "##a"]  # fmt: skip
    ids = {v: i for i, v in enumerate(dict.fromkeys(vocab))}
    ref = Tokenizer(models.WordPiece(ids, unk_token="[UNK]"))
    ref.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    for w in ["unafable", "casas", "cas", "xyz", "ca", "investigación", "inv"]:
        assert wordpiece(w, set(ids)) == ref.encode(w).tokens, w
    assert wordpiece("a" * 101, set(ids)) == ["[UNK]"]
    print("ej01 ok — igual que tokenizers.models.WordPiece")


if __name__ == "__main__":
    check()
```

`soluciones/ej02_mascara_relleno.py`:

```python
"""Ejercicio 2: máscara de relleno (padding) en la atención. Ver README.md."""

import numpy as np


def padding_mask(lengths: list[int], n: int) -> np.ndarray:
    """(lote, 1, n) booleana: True en las posiciones reales de cada secuencia."""
    return (np.arange(n)[None, :] < np.asarray(lengths)[:, None])[:, None, :]


def masked_attention(q, k, v, lengths):
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    scores = np.where(padding_mask(lengths, k.shape[-2]), scores, -np.inf)
    scores = scores - scores.max(-1, keepdims=True)
    w = np.exp(scores)
    return (w / w.sum(-1, keepdims=True)) @ v


def check() -> None:
    import torch

    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(3, 5, 4)) for _ in range(3))
    lengths = [5, 3, 1]
    mask = torch.tensor(padding_mask(lengths, 5))
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(torch.tensor(t) for t in (q, k, v)), attn_mask=mask
    )
    np.testing.assert_allclose(masked_attention(q, k, v, lengths), ref.numpy(), atol=1e-10)
    out = masked_attention(q, k, v, [1, 1, 1])
    np.testing.assert_allclose(out, np.broadcast_to(v[:, :1, :], out.shape), atol=1e-12)
    print("ej02 ok — igual que scaled_dot_product_attention con attn_mask")


if __name__ == "__main__":
    check()
```

`soluciones/ej03_posicional.py`:

```python
"""Ejercicio 3: codificación posicional sinusoidal (Vaswani et al., 2017). Ver README.md."""

import numpy as np


def sinusoidal_positions(n: int, d: int) -> np.ndarray:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(…); d par."""
    pos = np.arange(n)[:, None]
    freq = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return pe


def check() -> None:
    pe = sinusoidal_positions(50, 4)
    np.testing.assert_allclose(pe[0], [0, 1, 0, 1])
    np.testing.assert_allclose(pe[1], [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)])
    big = sinusoidal_positions(200, 64)
    # El producto punto solo depende de la distancia: <PE[p], PE[p+k]> no depende de p.
    for k in (1, 5, 20):
        dots = [big[p] @ big[p + k] for p in range(0, 150, 10)]
        np.testing.assert_allclose(dots, dots[0], atol=1e-9)
    print("ej03 ok — valores a mano y producto punto invariante a traslaciones")


if __name__ == "__main__":
    check()
```

`soluciones/ej04_tfidf.py`:

```python
"""Ejercicio 4: TF-IDF desde cero, igual a sklearn (idf suavizado, norma L2). Ver README.md."""

import re

import numpy as np

TOKEN = re.compile(r"(?u)\b\w+\b")


def tfidf(docs: list[str]) -> tuple[list[str], np.ndarray]:
    tokenized = [TOKEN.findall(d.lower()) for d in docs]
    vocab = sorted({t for doc in tokenized for t in doc})
    index = {t: i for i, t in enumerate(vocab)}
    tf = np.zeros((len(docs), len(vocab)))
    for r, doc in enumerate(tokenized):
        for t in doc:
            tf[r, index[t]] += 1
    df = (tf > 0).sum(0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    m = tf * idf
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return vocab, m / np.where(norms == 0, 1, norms)


def check() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = ["El aprendizaje por refuerzo", "aprendizaje profundo y refuerzo", "el cacao", ""]
    vocab, m = tfidf(docs)
    ref = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    r = ref.fit_transform(docs).toarray()
    assert vocab == list(ref.get_feature_names_out())
    np.testing.assert_allclose(m, r, atol=1e-12)
    print("ej04 ok — igual que TfidfVectorizer")


if __name__ == "__main__":
    check()
```

`soluciones/ej05_mrr_ndcg.py`:

```python
"""Ejercicio 5: MRR y NDCG con relevancia graduada. Ver README.md."""

import numpy as np


def mrr(rankings: list[list[str]], relevant: list[set[str]]) -> float:
    """Media de 1/posición del primer relevante (0 si no aparece)."""
    out = []
    for ranked, rel in zip(rankings, relevant, strict=True):
        out.append(next((1 / (i + 1) for i, r in enumerate(ranked) if r in rel), 0.0))
    return float(np.mean(out))


def ndcg_graded(ranked: list[str], gains: dict[str, float], k: int) -> float:
    """DCG = Σ gain / log2(pos + 1); el ideal ordena las ganancias de mayor a menor."""
    dcg = sum(gains.get(r, 0.0) / np.log2(i + 2) for i, r in enumerate(ranked[:k]))
    best = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(g / np.log2(i + 2) for i, g in enumerate(best))
    return float(dcg / ideal) if ideal > 0 else 0.0


def check() -> None:
    from sklearn.metrics import ndcg_score

    assert abs(mrr([["a", "b"], ["x", "y", "b"], ["z"]], [{"b"}, {"b"}, {"q"}]) - (1 / 2 + 1 / 3) / 3) < 1e-12
    rng = np.random.default_rng(0)
    for _ in range(100):
        ids = [f"d{i}" for i in range(12)]
        gains = {d: float(rng.integers(0, 4)) for d in ids}
        if not any(gains.values()):
            continue
        scores = rng.random(12)
        ranked = [ids[i] for i in np.argsort(-scores)]
        ref = ndcg_score([[gains[d] for d in ids]], [scores], k=5)
        assert abs(ndcg_graded(ranked, gains, 5) - ref) < 1e-12
    print("ej05 ok — MRR a mano y NDCG igual que sklearn")


if __name__ == "__main__":
    check()
```

`soluciones/ej06_refd.py`:

```python
"""Ejercicio 6: RefD en un grafo de enlaces pequeño. Ver README.md."""


def refd(a: str, b: str, links: dict[str, set[str]], concepts: set[str]) -> float:
    def refers(x, y):
        out = links.get(x, set()) & concepts
        if not out:
            return 0.0
        return sum(1 for c in out if c == y or y in links.get(c, set())) / len(out)

    return refers(a, b) - refers(b, a)


def check() -> None:
    import numpy as np

    from research_paths.prereq.refd import refd as reference

    links = {"Varianza": {"Media", "Suma"}, "Media": {"Suma"}, "Suma": set()}
    concepts = set(links)
    assert refd("Varianza", "Media", links, concepts) == 0.5
    assert refd("Media", "Suma", links, concepts) == 1.0
    rng = np.random.default_rng(0)
    names = [f"c{i}" for i in range(10)]
    for _ in range(20):
        g = {n: {m for m in names if m != n and rng.random() < 0.3} for n in names}
        for a in names:
            for b in names:
                assert refd(a, b, g, set(names)) == reference(a, b, g, set(names))
    print("ej06 ok — ejemplos a mano e igual a research_paths.prereq.refd")


if __name__ == "__main__":
    check()
```

Verificación a mano de ej06: Varianza enlaza {Media, Suma}; c = Media: c == Media ✓; c = Suma:
Suma no enlaza a Media → 1/2. Media enlaza {Suma}; Suma no enlaza a Varianza → 0. RefD = 0,5. ✓
Media → Suma: c = Suma == Suma → 1/1; Suma no enlaza a nada → 0. RefD = 1. ✓

`ejercicios/README.md`:

```markdown
# Ejercicios · PLN

Cada archivo tiene un `check()` que compara tu implementación con una referencia (`tokenizers`,
`torch`, scikit-learn o el propio paquete). Ejecuta `uv run python ejercicios/ej01_wordpiece.py`
desde la carpeta del módulo. Las soluciones están en `soluciones/`.

| # | Ejercicio | Qué practica |
|---|---|---|
| 1 | `ej01_wordpiece.py` | WordPiece greedy: el prefijo más largo y las continuaciones `##` |
| 2 | `ej02_mascara_relleno.py` | Máscara de relleno en la atención, contra `scaled_dot_product_attention` |
| 3 | `ej03_posicional.py` | Codificación posicional sinusoidal y por qué codifica distancias |
| 4 | `ej04_tfidf.py` | TF-IDF con idf suavizado y norma L2, igual a scikit-learn |
| 5 | `ej05_mrr_ndcg.py` | MRR y NDCG con relevancia graduada |
| 6 | `ej06_refd.py` | RefD: prerrequisitos a partir de enlaces |
```

- [ ] **Step 3: Stubs**

Para cada solución, crear `ejercicios/ejNN_*.py` con el mismo contenido pero el cuerpo de la(s)
función(es) a implementar reemplazado por `raise NotImplementedError` (en ej02, ambas funciones;
en ej05, ambas). Comprobar: `uv run python 02-semestre-2/02-pln/ejercicios/ej01_wordpiece.py`
falla con `NotImplementedError` (no con otro error).

- [ ] **Step 4: Verificar**

Run: `uv run pytest 02-semestre-2/02-pln/tests/test_soluciones.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git rm -q 02-semestre-2/02-pln/ejercicios/soluciones/.gitkeep
uv run ruff format && uv run ruff check --fix && uv run pytest
git add 02-semestre-2/02-pln/ejercicios 02-semestre-2/02-pln/tests/test_soluciones.py
git commit -m "docs(pln): seis ejercicios con soluciones verificadas contra referencias"
```

---

### Task 14: Ejecución real, informe y README del módulo

**Files:**
- Create: `M/proyecto/resultados/*` (de `run.py`), `M/proyecto/intro_etiquetas.csv`,
  `M/proyecto/resultados/demo_*.txt`
- Modify: `M/proyecto/README.md`, `M/README.md`, `README.md` (tabla de progreso),
  `M/notebooks/06-modelos-preentrenados.ipynb` (con salidas)

- [ ] **Step 1: Claves**

Run: `test -n "$OPENALEX_API_KEY" && test -n "$ANTHROPIC_API_KEY" && echo ok`
Expected: `ok`. Si falta alguna: **detenerse y pedirla al usuario** (no se puede seguir sin ella;
no crear claves ni pedirlas en texto plano: el usuario las exporta en su shell o en `.env`).

- [ ] **Step 2: Datos y ejecución completa**

```bash
uv run python 02-semestre-2/02-pln/data/download.py
uv run python 02-semestre-2/02-pln/proyecto/run.py > <workspace>/run.log 2>&1
```

Correr en segundo plano y esperar la notificación (≈ 1,5–2 h la primera vez: ~600 páginas de
Wikipedia, ~300 consultas a OpenAlex, ~6 000 llamadas al LLM, 9 ajustes de XLM-R). Al terminar,
leer el final de `run.log` y `resultados/metrics.json`.

Comprobar contra los hechos verificados: `capa1.cobertura.pairs == 6529`,
`pairs_unmapped + pairs_collision + pairs_kept == 6529`, `pairs_unmapped` ≈ 462 (6 529 − 6 067),
`concepts_unmapped` ≈ 40. Si difieren mucho, investigar antes de seguir (¿cambió Wikipedia?
¿fallaron peticiones?). `api.llm.unparsed` debe ser pequeño; anotarlo.

**Un resultado demasiado bueno se investiga** (p. ej. F1 del cross-encoder > 0,9 en un dominio no
visto: revisar que no haya fuga por conceptos compartidos o por títulos casi iguales).

- [ ] **Step 3: Etiquetar la muestra del clasificador introductorio**

Leer `resultados/intro_muestra.csv` (100 filas: id, tema, título, resumen, p_intro). **Sin mirar
`p_intro`**, etiquetar cada fila en `proyecto/intro_etiquetas.csv` con columnas `id,sirve`
(1 = sirve para empezar en el tema: revisión, tutorial, panorama o trabajo fundacional que
explica el tema sin presuponer el subcampo; 0 = contribución específica). Escribir ese criterio
en el README. Luego:

Run: `uv run python 02-semestre-2/02-pln/proyecto/run.py --acuerdo-intro`
Expected: imprime `{'n': 100, 'positivos': …, 'ap': …, 'f1@0,5': …}`.

- [ ] **Step 4: Demo y notebook 06**

```bash
for t in "Aprendizaje por refuerzo" "Serie temporal" "Moniliophthora roreri"; do
  uv run python -m research_paths ruta "$t" > "02-semestre-2/02-pln/proyecto/resultados/demo_$(echo $t | tr ' ' '_').txt"
done
uv run python scripts/run_notebooks.py --inplace 02-semestre-2/02-pln/notebooks/06-modelos-preentrenados.ipynb
```

Si un tema no resuelve (código 1), usar la sugerencia que imprime y registrar el título usado. Leer
las tres rutas: ¿tiene sentido el orden? Anotar en el README una cosa que salga bien y una que salga
mal de cada una (con la versión mínima de `order_route`).

- [ ] **Step 5: `proyecto/README.md`**

Reescribirlo con la estructura del de Visión: pregunta, sistema (diagrama de la spec §3), datos
(AL-CPL con cobertura real, Wikipedia, OpenAlex con temas por campo y fracción con resumen), capa 1
(tabla de F1 y AP por modelo; F1 por dominio; figura), capa 2 (tabla de recall@20/NDCG@20 por
orden, techo; figura; acuerdo del clasificador introductorio con su criterio), hipótesis (tabla
Δ, IC, p, p Holm, ¿se apoya?, con una viñeta de interpretación por hipótesis), demo (las tres
rutas resumidas), limitaciones (AL-CPL es de libro de texto y en inglés; etiqueta débil
review/article; ~44 % sin resumen; las referencias de una revisión no son «por dónde empezar»;
juez LLM no determinista entre versiones del modelo), uso (comandos) y **Tu turno**:
`order_route` en `src/research_paths/concepts.py`.

**Cada cifra sale de `metrics.json` o de los CSV de `resultados/`**: copiar con un script (no a
ojo) y releer el README contra el JSON antes del commit.

- [ ] **Step 6: README del módulo y tabla de progreso**

`M/README.md`: objetivos (5, como en Visión), tabla de temas → notebooks/ejercicios/proyecto,
estructura, cómo trabajar (`uv sync --extra base --extra dl --extra nlp`), recursos (Jurafsky &
Martin, *Speech and Language Processing* 3.ª ed. borrador; Vaswani et al. 2017; Sennrich et al.
2016 (BPE); Conneau et al. 2020 (XLM-R); Wang et al. 2022 (e5); Robertson & Zaragoza 2009 (BM25);
Liang et al. 2015 (RefD) y 2018 (AL-CPL); Priem et al. 2022 (OpenAlex)), checklist con todo en
[x] salvo «CI en verde».

`README.md` raíz, fila de PLN: mini-proyecto «Motor de rutas de investigación (Wikipedia +
OpenAlex)», estado 🚧 (pasa a ✅ cuando el CI esté verde tras el merge).

- [ ] **Step 7: Commit**

```bash
uv run ruff format && uv run ruff check --fix && uv run pytest && uv run python scripts/run_notebooks.py
git add 02-semestre-2/02-pln README.md
git commit -m "docs(pln): resultados del motor de rutas, informe del proyecto y README del módulo"
```

---

## Cierre

- Revisión final de toda la rama (revisor en el modelo más capaz) con el foco de «Review Focus».
- Preguntar al usuario antes del merge a `main`. Tras el merge: push, seguir el CI, marcar ✅ en el
  README raíz y en el checklist, actualizar la memoria del repo (módulo PLN ✅ y TODO `order_route`).
