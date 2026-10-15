# Solución ejercicio 5: estabilidad de GD en cuadráticas

Sea $H = Q\Lambda Q^\top$ y $z = Q^\top x$. GD da $x_{k+1} = (I - \eta H)x_k$, así que en la
base propia cada coordenada evoluciona de forma independiente:

$$z_{k+1}^{(i)} = (1 - \eta\lambda_i)\, z_k^{(i)} \quad\Rightarrow\quad z_k^{(i)} = (1-\eta\lambda_i)^k z_0^{(i)}.$$

**Convergencia desde todo $x_0$** $\iff$ $|1 - \eta\lambda_i| < 1$ para todo $i$
$\iff$ $0 < \eta\lambda_i < 2$ $\iff$ $0 < \eta < 2/\lambda_{max}$
(la condición más restrictiva es la del autovalor mayor).

**Contracción óptima.** La tasa es $r(\eta) = \max_i |1 - \eta\lambda_i| =
\max(|1-\eta\lambda_{min}|, |1-\eta\lambda_{max}|)$, pues $|1-\eta\lambda|$ es convexa en $\lambda$
y su máximo en un intervalo se alcanza en un extremo. Se minimiza cuando ambos extremos se equilibran:

$$1 - \eta\lambda_{min} = \eta\lambda_{max} - 1 \;\Rightarrow\; \eta^* = \frac{2}{\lambda_{min}+\lambda_{max}},
\qquad r^* = \frac{\lambda_{max}-\lambda_{min}}{\lambda_{max}+\lambda_{min}} = \frac{\kappa-1}{\kappa+1}.$$

Para reducir el error en un factor $\varepsilon$ se necesitan
$k \approx \frac{\log(1/\varepsilon)}{-\log r^*} \approx \frac{\kappa}{2}\log(1/\varepsilon)$
iteraciones: lineal en $\kappa$, lo que se observa empíricamente en el notebook 03.
