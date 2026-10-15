# Solución ejercicio 5 (papel)

**(a) ALS explícito.** Con $q$ fijo, para el usuario $u$ con ítems calificados $I_u$:
$\min_{p} \sum_{i \in I_u} (r_{ui} - q_i^\top p)^2 + \lambda n_u \|p\|^2$. Es ridge: igualando el
gradiente a 0, $(Q_u^\top Q_u + \lambda n_u I)\,p_u = Q_u^\top r_u$, con $Q_u$ las filas de los ítems de
$u$. Cada usuario es independiente (paralelizable) y la pérdida total no aumenta en ningún paso, porque
cada actualización es un mínimo exacto (test `test_als_loss_is_monotone`).

**(b) ALS implícito ≠ PureSVD.** Ambos ponen 0 donde no hay interacción, pero PureSVD minimiza el
error cuadrático **sin pesos**: un 0 “no lo vi” pesa igual que un 1 “lo vi”. ALS implícito pondera con
la confianza $c_{ui} = 1 + \alpha r_{ui}$: los ceros pesan 1 y las interacciones $1 + \alpha r$. Con α → 0
se acerca a una factorización sin pesos (como PureSVD, pero con regularización); con α grande, los
ceros casi no cuentan. El notebook 03 muestra el efecto de α en NDCG y cobertura.

**(c) NDCG.** $\text{DCG}@k = \sum_{p=1}^{k} \text{rel}_p / \log_2(p+1) \ge 0$. Con relevancia binaria,
el ideal pone todos los relevantes posibles arriba: $\text{IDCG} = \sum_{p=1}^{\min(k,|\text{rel}|)} 1/\log_2(p+1)$.
Como los descuentos son estrictamente decrecientes, cualquier lista con *m* aciertos suma a lo más los
*m* primeros descuentos, así que $\text{DCG} \le \text{IDCG}$, con igualdad sii los aciertos ocupan los
primeros $\min(k, |\text{rel}|)$ puestos.

**(d) Pregunta abierta.** Hipótesis para discutir y contrastar, no respuestas:
1. *Saturación:* con ~40 relevantes por usuario, casi cualquier lista razonable acierta, y las
   diferencias entre modelos se comprimen (mirar recall además de NDCG).
2. *Exposición temporal:* los usuarios activos califican durante más tiempo; películas populares
   estrenadas durante ese periodo caen en su test (la partición es temporal). Contrastar
   restringiendo a películas ya existentes al inicio del historial de cada usuario.
3. *Modelos sesgados hacia el gusto pasado:* los factores de un usuario con historia larga reflejan
   sus gustos antiguos; la popularidad reciente captura cambios. Contrastar ponderando la historia
   por recencia.
