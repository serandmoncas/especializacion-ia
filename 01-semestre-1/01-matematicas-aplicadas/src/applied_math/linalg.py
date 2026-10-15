"""Álgebra lineal aplicada: proyecciones, autovalores y PCA."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def project_onto(v: Array, A: Array) -> Array:
    """Proyección ortogonal de `v` sobre el espacio columna de `A`.

    P = A (AᵀA)⁻¹ Aᵀ. Se resuelve con `lstsq` en lugar de invertir AᵀA,
    que es numéricamente inestable cuando A está mal condicionada.
    """
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return A @ coef


def power_iteration(
    M: Array, *, n_iter: int = 1000, tol: float = 1e-10, rng: np.random.Generator | None = None
) -> tuple[float, Array]:
    """Autovalor dominante (en magnitud) y su autovector, por iteración de potencias.

    Converge a razón |λ₂/λ₁| por iteración: lento si los dos primeros
    autovalores son parecidos.
    """
    rng = rng or np.random.default_rng(0)
    v = rng.normal(size=M.shape[0])
    v /= np.linalg.norm(v)
    eigval = 0.0
    for _ in range(n_iter):
        w = M @ v
        v_next = w / np.linalg.norm(w)
        eigval_next = float(v_next @ M @ v_next)  # cociente de Rayleigh
        if abs(eigval_next - eigval) < tol:
            return eigval_next, v_next
        v, eigval = v_next, eigval_next
    return eigval, v


def pca(X: Array, n_components: int) -> tuple[Array, Array, Array]:
    """PCA vía SVD de los datos centrados.

    Devuelve (proyecciones, componentes [k, d], varianza explicada [k]).
    Usar SVD sobre X evita formar XᵀX, que eleva al cuadrado el número de condición.
    """
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    explained_variance = S[:n_components] ** 2 / (X.shape[0] - 1)
    return Xc @ components.T, components, explained_variance
