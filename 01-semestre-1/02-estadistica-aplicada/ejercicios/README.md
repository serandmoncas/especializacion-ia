# Ejercicios — Estadística aplicada

Cada ejercicio tiene un archivo `ejNN_*.py` con firmas, docstrings y una función `check()`.
Implementar y ejecutar `uv run python ejercicios/ejNN_*.py` desde la carpeta del módulo.
Las soluciones están en [`soluciones/`](soluciones/): intentarlo antes de mirarlas.

| # | Tema | Archivo |
|---|---|---|
| 1 | Intervalos para proporciones: Wald vs. Wilson | `ej01_wilson.py` |
| 2 | Bootstrap BCa desde cero | `ej02_bca.py` |
| 3 | Fronteras de Pocock para mirar resultados sin inflar α | `ej03_pocock.py` |
| 4 | Thompson sampling: de A/B test a bandido | `ej04_thompson.py` |
| 5 | (Papel) Derivación de CUPED | abajo |

## 1. Wald vs. Wilson

El intervalo de Wald $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ es el de los libros y se rompe con
$p$ pequeño: si $\hat p = 0$ el intervalo tiene ancho cero. El de Wilson invierte la prueba
score: es el conjunto de $p$ con $|\hat p - p| \le z\sqrt{p(1-p)/n}$. Resolver esa cuadrática
en $p$, implementar ambos y medir su cobertura con $n = 50$, $p = 0.02$.

## 2. BCa

El percentil asume que la distribución bootstrap está centrada y tiene forma constante.
BCa corrige con dos constantes: sesgo $z_0 = \Phi^{-1}(\#\{\hat\theta^* < \hat\theta\}/B)$ y
aceleración $a$ (estimada por jackknife). Los cuantiles a usar son
$\alpha_j = \Phi\big(z_0 + \frac{z_0 + z_{\alpha_j}}{1 - a(z_0 + z_{\alpha_j})}\big)$.
Implementar y comparar con `scipy.stats.bootstrap(method="BCa")`.

## 3. Pocock

En el notebook 02, mirar 20 veces con umbral 0.05 da ~24 % de falsos positivos. Pocock usa un
umbral **constante** $c$ en cada mirada, elegido para que $P(\max_k |Z_k| > c) = 0.05$ bajo $H_0$.
Con $K$ miradas equiespaciadas, $Z_k = S_k/\sqrt{k}$ con $S_k$ suma de $k$ normales estándar.
Encontrar $c$ por simulación + bisección para $K = 5$ (valor de tablas: 2.413).

## 4. Thompson sampling

Un A/B test reparte tráfico 50/50 hasta el final; un bandido lo desplaza hacia el brazo que
va ganando. En cada ronda: muestrear $p_i \sim \text{Beta}$ posterior de cada brazo, jugar el
mayor, actualizar. Comparar el *regret* (conversiones perdidas frente a jugar siempre el mejor)
con el reparto 50/50. ¿Qué se pierde a cambio? (Pista: la estimación del brazo malo.)

## 5. CUPED (papel)

Sea $Y_{adj} = Y - \theta(X - \bar X)$ con $X$ previa al experimento.
(a) Mostrar que $E[\bar Y_{adj,T} - \bar Y_{adj,C}] = E[\bar Y_T - \bar Y_C]$ si la asignación es aleatoria.
(b) Mostrar que $\theta^* = \text{Cov}(Y,X)/\text{Var}(X)$ minimiza $\text{Var}(Y_{adj})$ y que
$\text{Var}(Y_{adj}) = \text{Var}(Y)(1-\rho^2)$.
(c) En Hillstrom, $\rho(\text{spend}, \text{history}) = 0.022$. ¿Cuánto tamaño de muestra
“ahorra” CUPED? ¿Qué $\rho$ haría falta para reducir el $n$ necesario a la mitad?
