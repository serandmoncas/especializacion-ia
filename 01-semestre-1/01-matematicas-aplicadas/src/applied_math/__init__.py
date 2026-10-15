"""Matemáticas aplicadas: álgebra lineal, gradientes y optimización desde cero en NumPy."""

from applied_math.gradients import gradient_check, numerical_gradient
from applied_math.linalg import pca, power_iteration, project_onto
from applied_math.linear_regression import (
    LinearRegressionGD,
    fit_normal_equation,
    mse,
    mse_gradient,
)
from applied_math.optim import OptimResult, gradient_descent, minibatch_sgd

__all__ = [
    "LinearRegressionGD",
    "OptimResult",
    "fit_normal_equation",
    "gradient_check",
    "gradient_descent",
    "minibatch_sgd",
    "mse",
    "mse_gradient",
    "numerical_gradient",
    "pca",
    "power_iteration",
    "project_onto",
]
