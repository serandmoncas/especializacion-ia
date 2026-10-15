# Solución ejercicio 5: CUPED

**(a) Insesgadez.** $\bar Y_{adj,g} = \bar Y_g - \theta(\bar X_g - \bar X)$, así que
$\bar Y_{adj,T} - \bar Y_{adj,C} = (\bar Y_T - \bar Y_C) - \theta(\bar X_T - \bar X_C)$.
$X$ se mide antes del tratamiento y la asignación es aleatoria, luego $E[\bar X_T] = E[\bar X_C]$
y el segundo término tiene esperanza cero. (Si $\theta$ se estima con los datos, el sesgo es
$O(1/n)$; estimarlo con los grupos combinados evita que dependa del tratamiento.)

**(b) θ óptimo.**
$\text{Var}(Y - \theta X) = \text{Var}(Y) - 2\theta\,\text{Cov}(Y,X) + \theta^2\text{Var}(X)$,
una parábola en $\theta$ con mínimo en $\theta^* = \text{Cov}(Y,X)/\text{Var}(X)$
(el coeficiente de regresión de $Y$ sobre $X$). Sustituyendo:

$$\text{Var}(Y_{adj}) = \text{Var}(Y) - \frac{\text{Cov}(Y,X)^2}{\text{Var}(X)} = \text{Var}(Y)(1 - \rho^2).$$

**(c) Hillstrom.** Como $n \propto \text{Var}$ para una potencia fija, CUPED multiplica el $n$
necesario por $1 - \rho^2 = 1 - 0.022^2 \approx 0.9995$: ahorra 0.05 %, nada.
Para reducir $n$ a la mitad hace falta $\rho^2 = 0.5$, es decir $\rho \approx 0.71$.
En producción eso se consigue usando **la misma métrica en el período previo** (gasto de las
2 semanas anteriores), no un agregado anual; el proyecto lo confirma: ni todas las covariables
juntas explican más que $R^2 = 0.0009$ del gasto.
