"""Ejecuta el `check()` de cada solución: garantiza que las soluciones publicadas son correctas."""

import importlib.util
from pathlib import Path

import pytest

SOLUTIONS = sorted((Path(__file__).parents[1] / "ejercicios" / "soluciones").glob("ej*.py"))


def test_there_are_six_solutions():
    assert len(SOLUTIONS) == 6


@pytest.mark.parametrize("path", SOLUTIONS, ids=lambda p: p.stem)
def test_solution_passes_its_check(path):
    pytest.importorskip("skimage")
    spec = importlib.util.spec_from_file_location(f"sol_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check()
