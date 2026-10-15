# Solución ejercicio 5 (papel)

**(a) Softmax + entropía cruzada.** $\text{CE} = -\log p_y = -z_y + \log\sum_j e^{z_j}$. Derivando:
$\partial \text{CE}/\partial z_i = -[i = y] + e^{z_i}/\sum_j e^{z_j} = p_i - [i = y]$.
Por eso las librerías fusionan softmax y CE: el gradiente conjunto es simple y estable, mientras
que componerlos por separado divide por $p_y$, que puede ser casi 0.

**(b) Inicialización He.** Con $z = Wx$, $E[z_i^2] = \text{fan\_in}\,\sigma^2 E[x^2]$. La ReLU anula la
mitad negativa de una $z$ simétrica: $E[h^2] = \tfrac12 E[z^2]$. Para que $E[h^2]$ no cambie entre
capas hace falta $\tfrac12 \text{fan\_in}\,\sigma^2 = 1$, es decir $\sigma^2 = 2/\text{fan\_in}$.
Con $\sigma = 1$ (notebook 01) $E[h^2]$ se multiplica por $\text{fan\_in}/2 = 64$ en cada capa, es decir la
norma por $\sqrt{64} = 8$: tras 20 capas, $8^{20} \approx 10^{18}$, el orden que muestra la figura.

**(c) LSTM.** Ignorando la dependencia de las compuertas respecto de $c$,
$\partial c_t / \partial c_{t-k} = \prod_{s=t-k+1}^{t} \text{diag}(f_s)$. Con $f \approx 0.5$
(sesgo 0, inicialización por defecto) el producto es $\approx 0.5^k$: desvanecimiento igual que en
una RNN. Con $f \approx \sigma(3) \approx 0.95$, $0.95^{100} \approx 0.006$ frente a $0.5^{100} \approx 10^{-30}$.
Es lo que separa las curvas del notebook 04. A diferencia de la RNN, no hay multiplicación por
$W_h$ ni por la derivada de tanh en ese camino.

**(d) Parámetros de la CNN del proyecto.** Conv: $(k^2 c_{in} + 1)\,c_{out}$; BN: $2c$ (γ, β).

| Capa | Cálculo | Parámetros |
|---|---|---|
| conv 1→32 | (9·1 + 1)·32 | 320 |
| BN 32 | 2·32 | 64 |
| conv 32→32 | (9·32 + 1)·32 | 9 248 |
| BN 32 | | 64 |
| conv 32→64 | (9·32 + 1)·64 | 18 496 |
| BN 64 | | 128 |
| conv 64→64 | (9·64 + 1)·64 | 36 928 |
| BN 64 | | 128 |
| denso 3136→128 | 3136·128 + 128 | 401 536 |
| denso 128→10 | 128·10 + 10 | 1 290 |
| **Total** | | **468 202** |

El 86 % está en la primera capa densa: las convoluciones son baratas, aplanar un mapa 64×7×7 no.
Reemplazarla por *global average pooling* (promediar cada canal de 7×7 a 1 valor) seguido directamente
de la capa de 10 clases dejaría el modelo en 65 376 + 650 = 66 026 parámetros.
