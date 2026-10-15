# Solución ejercicio 5 (papel)

**(a) Caminata aleatoria.** $y_{T+h} = y_T + \sum_{j=1}^{h} \varepsilon_{T+j}$. Dada la historia,
$E[y_{T+h} \mid y_{1:T}] = y_T$: el naive es el pronóstico óptimo en error cuadrático. El error es
la suma de $h$ ruidos independientes, con varianza $h\sigma^2$: la incertidumbre crece como
$\sqrt h$ y el intervalo se abre como un embudo.

**(b) MASE < 1.** El denominador es el MAE *in-sample* del seasonal naive (errores a un ciclo de
distancia sobre el entrenamiento), no su MAE en el test. Si el test es más predecible que el
entrenamiento (o el horizonte es distinto), el seasonal naive puede tener en el test un MAE menor
que su escala, y un modelo con MASE 0.9 puede aun así perder contra él. Para afirmar “le gana al
baseline” hay que evaluar el baseline en el mismo test, como hace el proyecto.

**(c) SES ≡ ARIMA(0,1,1).** SES: $\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t$, o sea
$\hat y_{t+1} = \hat y_t + \alpha e_t$ con $e_t = y_t - \hat y_t$. Entonces
$y_{t+1} = \hat y_{t+1} + e_{t+1} = y_t - e_t + \alpha e_t + e_{t+1}$, es decir
$\nabla y_{t+1} = e_{t+1} + (\alpha - 1) e_t$: un ARIMA(0,1,1) con $\theta = \alpha - 1$, cuyos
errores son los errores a un paso del SES.
