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
