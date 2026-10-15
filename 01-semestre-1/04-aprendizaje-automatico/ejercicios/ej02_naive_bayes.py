"""Ejercicio 2: naive Bayes gaussiano en espacio logarítmico. Ver README.md."""

import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        raise NotImplementedError

    def joint_log_likelihood(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_log_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError


def check() -> None:
    from sklearn.datasets import load_digits, make_classification
    from sklearn.naive_bayes import GaussianNB

    X, y = load_digits(return_X_y=True)
    ours = GaussianNaiveBayes().fit(X, y)
    ref = GaussianNB().fit(X, y)
    np.testing.assert_allclose(ours.predict_log_proba(X), ref.predict_log_proba(X), atol=1e-6)
    assert np.array_equal(ours.predict(X), ref.predict(X))
    # La log-verosimilitud escala como ~ −1.4·d y exp() subdesborda bajo −745: con d = 1000,
    # multiplicar densidades da 0 para todas las clases y la clasificación es imposible.
    Xh, yh = make_classification(500, n_features=1000, n_informative=20, random_state=0)
    jll = GaussianNaiveBayes().fit(Xh, yh).joint_log_likelihood(Xh)
    underflow = np.mean(np.exp(jll).max(axis=1) == 0)
    assert underflow > 0.9
    print(f"ej02 ok — coincide con sklearn; con d = 1000, sin logs {underflow:.0%} de filas da 0")


if __name__ == "__main__":
    check()
