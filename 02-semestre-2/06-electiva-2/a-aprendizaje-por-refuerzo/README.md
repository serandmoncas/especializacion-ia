# Aprendizaje por refuerzo

> Electiva 2 (S2) · **Estado: plantilla, sin desarrollar**

## Temas

- Procesos de decisión de Markov y programación dinámica
- Monte Carlo y diferencias temporales (Q-learning, SARSA)
- Aproximación de funciones y DQN
- Gradiente de política

## Mini-proyecto

Agente que aprende a controlar un entorno clásico, comparado con una política base. Ver [`proyecto/`](proyecto/).

## Cómo trabajar el módulo

```bash
uv sync --extra base --extra dl
uv run pytest 02-semestre-2/06-electiva-2/a-aprendizaje-por-refuerzo
```

## Checklist (definición de terminado)

- [ ] README con objetivos, temas y recursos
- [ ] Al menos 2 notebooks ejecutables sin errores
- [ ] Ejercicios con soluciones en carpeta aparte
- [ ] Mini-proyecto con README, código y resultados
- [ ] Tests de las funciones en `src/`
- [ ] CI en verde
