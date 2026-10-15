# Solución ejercicio 5: admisibilidad y consistencia

**(a) Consistente ⇒ admisible.** Sea $n = n_0, n_1, \dots, n_k$ = meta un camino óptimo.
Aplicando consistencia en cada arista y sumando (telescópica):
$h(n_0) \le \sum_{i} c(n_i, n_{i+1}) + h(n_k) = h^*(n) + 0$.

**(b) Admisible no consistente.** Grafo $A \to B \to G$ con costos 1 y 1; $h(A) = 2$, $h(B) = 0$,
$h(G) = 0$. Es admisible ($h^*(A) = 2$, $h^*(B) = 1$), pero $h(A) = 2 > c(A,B) + h(B) = 1$.
Con heurísticas así, A* puede cerrar un nodo con un $g$ no óptimo y necesita reabrirlo
(por eso `best_first` admite `reopen=True`).

**(c) f no decreciente.** Si $n'$ es sucesor de $n$:
$f(n') = g(n) + c(n,n') + h(n') \ge g(n) + h(n) = f(n)$.
A* expande en orden de $f$ no decreciente; cuando expande $n$, cualquier otro camino a $n$
pasa por un nodo de la frontera con $f \ge f(n)$ y por lo tanto llega con $g \ge g(n)$:
el primer $g$ con que se expande $n$ es óptimo y nunca hace falta reabrirlo.
(El test `test_astar_with_consistent_heuristic_never_reexpands` lo verifica empíricamente.)

**(d) Octile es consistente.** Sea $h(x,y) = \max(d_x, d_y) + (\sqrt2 - 1)\min(d_x, d_y)$
con $d_x = |x - x_g|$, $d_y = |y - y_g|$. Un paso cardinal cambia una sola coordenada en 1, y $h$
cambia a lo sumo en $\max(1, \sqrt2 - 1) = 1 = c$. Un paso diagonal cambia ambas en 1, y $h$
cambia a lo sumo en $1 + (\sqrt2 - 1) = \sqrt2 = c$. Es decir, $h$ es 1-Lipschitz respecto al
costo de cada movimiento: $|h(n) - h(n')| \le c(n,n')$, que implica consistencia. Los obstáculos
solo eliminan aristas, así que la desigualdad sigue valiendo en el grafo restringido.

**(e)** $\max(h_1, h_2) \le \max(h^*, h^*) = h^*$: admisible, y domina a cada una (además,
el máximo de consistentes es consistente). $h_1 + h_2$ en general **no**: con $h_1 = h_2 = h^*$
da $2h^*$. Excepción: heurísticas *aditivas* que cuentan costos de subconjuntos disjuntos de
movimientos (bases de datos de patrones disjuntas en el 15-puzzle).
