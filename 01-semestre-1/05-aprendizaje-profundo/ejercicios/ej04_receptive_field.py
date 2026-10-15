"""Ejercicio 4: campo receptivo de una CNN. Ver README.md."""


def receptive_field(layers: list[tuple[int, int]]) -> list[int]:
    """layers: (kernel, stride) en orden. Devuelve el campo receptivo tras cada capa."""
    raise NotImplementedError


# CNN del proyecto: [conv3, conv3, pool2] × 2
PROJECT_CNN = [(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2)]


def empirical_receptive_field(seed: int = 0) -> int:
    """Ancho de la región de entrada con gradiente ≠ 0 para una unidad central de la última conv."""
    raise NotImplementedError


def check() -> None:
    rf = receptive_field(PROJECT_CNN)
    assert rf == [3, 5, 6, 10, 14, 16], rf
    # hasta la última conv (sin el pool final): 14
    emp = empirical_receptive_field()
    assert emp == rf[4], (emp, rf[4])
    print(f"ej04 ok — campo receptivo por capa {rf}; verificado con gradientes: {emp} pixeles")


if __name__ == "__main__":
    check()
