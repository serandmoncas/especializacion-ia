# Aprendizaje profundo

> Semestre 1 · 2 créditos · Compartida · requiere `uv sync --extra base --extra dl`

## Objetivos

Al terminar el módulo se debe poder:

1. Explicar backpropagation como regla de la cadena sobre un grafo e implementar un autograd.
2. Diagnosticar redes que no entrenan: inicialización, activaciones, tasa de aprendizaje, gradientes.
3. Regularizar (weight decay, dropout, early stopping, aumentación) y medir el efecto en validación.
4. Construir CNN y RNN en PyTorch y compararlas con baselines de forma estadísticamente honesta.

## Temas

| Tema | Dónde |
|---|---|
| Backprop, modo reverso, gradientes que se desvanecen/explotan, inicialización | `notebooks/01-autograd-y-backprop.ipynb` |
| SGD/momentum/Adam, tasa de aprendizaje, sobreajuste, weight decay, dropout | `notebooks/02-optimizacion-y-regularizacion.ipynb` |
| Convolución, pesos compartidos, CNN vs. MLP, robustez a desplazamientos | `notebooks/03-convoluciones.ipynb` |
| RNN, LSTM, GRU, memoria, sesgo de compuertas, gradientes en el tiempo | `notebooks/04-redes-recurrentes.ipynb` |
| Atención, batch norm, max pooling, softmax estable, conteo de parámetros | `ejercicios/` |
| CNN en Fashion-MNIST frente a baselines | `proyecto/` |

## Estructura

```
05-aprendizaje-profundo/
├── notebooks/            # 01 autograd · 02 optimización · 03 CNN · 04 RNN
├── src/applied_dl/
│   ├── autograd.py       # Tensor con diferenciación automática en modo reverso (NumPy)
│   ├── nn.py             # Linear, MLP, Dropout, inicialización, SGD, Adam/AdamW
│   ├── conv.py           # conv2d por im2col, con backward
│   └── training.py       # bucle de entrenamiento PyTorch con early stopping
├── tests/                # gradientes vs. numéricos y vs. torch; optimizadores paso a paso vs. torch.optim
├── ejercicios/           # soluciones/ aparte
├── data/download.py      # Fashion-MNIST vía torchvision
└── proyecto/             # 5 modelos + McNemar
```

## Cómo trabajar el módulo

```bash
uv sync --extra base --extra dl
uv run jupyter lab 01-semestre-1/05-aprendizaje-profundo/notebooks
uv run pytest 01-semestre-1/05-aprendizaje-profundo
```

En Linux (y en el CI) `torch` se instala solo-CPU desde el índice de PyTorch; en macOS, desde PyPI
con soporte MPS. Ver `[tool.uv.sources]` en `pyproject.toml`.

## Recursos

- I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning* (2016): caps. 6–10. Libre en deeplearningbook.org.
- S. Prince, *Understanding Deep Learning* (2023): libre en udlbook.github.io; muy buenas figuras.
- A. Karpathy, *micrograd* y “Neural Networks: Zero to Hero” (video): el autograd de este módulo
  sigue la misma idea, extendida a tensores y broadcasting.
- K. He et al., “Delving Deep into Rectifiers” (2015): inicialización He.
- R. Jozefowicz, W. Zaremba, I. Sutskever, “An Empirical Exploration of Recurrent Network
  Architectures” (ICML 2015): el sesgo de la compuerta de olvido.
- I. Loshchilov, F. Hutter, “Decoupled Weight Decay Regularization” (ICLR 2019): AdamW.

## Checklist (definición de terminado)

- [x] README con objetivos, temas y recursos
- [x] Al menos 2 notebooks ejecutables sin errores (4, marcados `smoke`)
- [x] Ejercicios con soluciones en carpeta aparte (5; soluciones verificadas contra torch)
- [x] Mini-proyecto con README, código y resultados (aumentación: tu turno)
- [x] Tests de las funciones en `src/`
- [ ] CI en verde
