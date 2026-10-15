# IA en el borde

> Electiva 2 (S2) · **Estado: plantilla, sin desarrollar**

## Temas

- Restricciones de dispositivo: memoria, latencia, energía
- Cuantización y poda
- Destilación de conocimiento
- Exportación (ONNX) y medición en CPU

## Mini-proyecto

Comprimir un modelo del semestre manteniendo su precisión, con medición de latencia y tamaño. Ver [`proyecto/`](proyecto/).

## Cómo trabajar el módulo

```bash
uv sync --extra base --extra dl
uv run pytest 02-semestre-2/06-electiva-2/b-ia-en-el-borde
```

## Checklist (definición de terminado)

- [ ] README con objetivos, temas y recursos
- [ ] Al menos 2 notebooks ejecutables sin errores
- [ ] Ejercicios con soluciones en carpeta aparte
- [ ] Mini-proyecto con README, código y resultados
- [ ] Tests de las funciones en `src/`
- [ ] CI en verde
