# Ejercicios — Fundamentos de IA

Cada ejercicio tiene un archivo `ejNN_*.py` con firmas, docstrings y una función `check()`.
Implementar y ejecutar `uv run python ejercicios/ejNN_*.py` desde la carpeta del módulo.
Las soluciones están en [`soluciones/`](soluciones/): intentarlo antes de mirarlas.

| # | Tema | Archivo |
|---|---|---|
| 1 | Satisfacción de restricciones: N reinas con MRV y forward checking | `ej01_nqueens.py` |
| 2 | Búsqueda adversaria: minimax y poda alfa-beta en tres en raya | `ej02_alphabeta.py` |
| 3 | Búsqueda bidireccional | `ej03_bidirectional.py` |
| 4 | Resolución proposicional | `ej04_resolution.py` |
| 5 | (Papel) Admisibilidad y consistencia | abajo |

## 1. N reinas como CSP

Variables: columnas; dominio: filas; restricciones: ninguna pareja se ataca. Implementar
backtracking (a) ingenuo, en orden de columnas, y (b) con **MRV** (elegir la variable con menos
valores legales) y **forward checking** (al asignar, podar los dominios de las no asignadas).
Comparar nodos visitados para N = 8, 20, 50.

## 2. Alfa-beta

Implementar minimax y minimax con poda alfa-beta para el tres en raya. Verificar que el juego
perfecto desde el tablero vacío es empate y contar nodos: el árbol completo de minimax tiene
549 946 nodos; ¿cuántos visita alfa-beta? ¿Y si se ordenan los movimientos (centro, esquinas, lados)?

## 3. Bidireccional

Dos BFS simultáneas, desde el inicio y desde la meta, que se detienen al encontrarse.
Con factor de ramificación $b$ y profundidad $d$ explora $O(b^{d/2})$ en vez de $O(b^d)$.
Implementarla para grillas 4-conectadas (costo unitario) y comparar expansiones con BFS.
¿Qué hace falta para que el camino devuelto sea óptimo? (Pista: expandir capas completas.)
¿Por qué en una grilla abierta el ahorro es mucho menor que el que promete $O(b^{d/2})$?

## 4. Resolución

$KB \models \alpha$ sii $KB \wedge \neg\alpha$ es insatisfacible. Con la base en CNF (cláusulas
como conjuntos de literales), resolver pares de cláusulas hasta derivar la cláusula vacía
(⇒ implicación demostrada) o no generar nada nuevo (⇒ no se implica). Validar contra
`tt_entails` en el mundo del wumpus del notebook 04.

## 5. Admisibilidad y consistencia (papel)

(a) Demostrar que *consistente* ($h(n) \le c(n, n') + h(n')$, $h(meta) = 0$) implica *admisible*.
(b) Dar un ejemplo de heurística admisible pero no consistente.
(c) Demostrar que con $h$ consistente, $f = g + h$ es no decreciente a lo largo de cualquier
camino, y concluir que A* nunca necesita reabrir un nodo.
(d) Demostrar que octile es consistente en una grilla 8-conectada sin cortar esquinas.
(e) Si $h_1, h_2$ son admisibles, ¿lo es $\max(h_1, h_2)$? ¿Y $h_1 + h_2$?
