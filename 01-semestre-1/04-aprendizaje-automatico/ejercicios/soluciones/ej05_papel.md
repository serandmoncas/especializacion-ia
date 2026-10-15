# Solución ejercicio 5 (papel)

**(a) Sesgo–varianza.** Sea $y = f(x) + \varepsilon$, $E[\varepsilon] = 0$, $\text{Var}(\varepsilon)=\sigma^2$,
y $\hat f$ aleatoria (depende del conjunto de entrenamiento) e independiente de $\varepsilon$.
Con $\bar f = E[\hat f(x)]$:

$$E[(y-\hat f)^2] = E[(f + \varepsilon - \bar f + \bar f - \hat f)^2]
= \sigma^2 + (f - \bar f)^2 + E[(\hat f - \bar f)^2],$$

porque los productos cruzados tienen esperanza cero ($E[\varepsilon]=0$, $E[\hat f - \bar f]=0$ y
$f - \bar f$ es constante).

**(b) AUC = P(s⁺ > s⁻).** La ROC recorre umbrales $t$: $\text{TPR}(t) = P(s^+ > t)$,
$\text{FPR}(t) = P(s^- > t)$. Entonces
$\text{AUC} = \int \text{TPR}\, d\text{FPR} = \int P(s^+ > t)\, f_{s^-}(t)\, dt = P(s^+ > s^-)$.
Con puntajes discretos, la regla del trapecio sobre un tramo de empates aporta la mitad de los
pares empatados: $+\tfrac12 P(s^+ = s^-)$. Contar pares es el estadístico U de Mann–Whitney
dividido por $n_+ n_-$, que es lo que calcula `applied_ml.metrics.roc_auc` con rangos.

**(c) Convexidad.** Por observación, $\ell(z) = \log(1+e^z) - yz$ con $z = w^\top x$.
$\ell''(z) = \sigma(z)(1-\sigma(z)) \ge 0$, así que $\nabla^2_w \ell = \sigma(1-\sigma)\, x x^\top \succeq 0$.
La suma de convexas es convexa, y $+\tfrac{\alpha}{2}\|w\|^2$ la hace fuertemente convexa
(mínimo único): por eso Newton converge desde $w = 0$.

**(d) OOB.** Una muestra concreta no es elegida en un sorteo con probabilidad $1 - 1/n$; en $n$
sorteos independientes, $(1-1/n)^n \to e^{-1} \approx 0.368$. Cada árbol deja fuera ~36.8 % de los
datos, y con 100 árboles cada punto queda fuera de ~37 de ellos: suficiente para promediar una
predicción OOB.
