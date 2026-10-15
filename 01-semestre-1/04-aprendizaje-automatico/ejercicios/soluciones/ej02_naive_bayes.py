"""Solución ejercicio 2: naive Bayes gaussiano.

log P(y|x) = log P(y) + Σ_j log N(x_j; μ_yj, σ²_yj) − log Z. Sumar logs evita el underflow de
multiplicar densidades; log Z se calcula con log-sum-exp (restando el máximo antes de exp).
var_smoothing suma ε·max(var) a todas las varianzas, para que ninguna sea 0.
"""

import numpy as np
from scipy.special import logsumexp


class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        eps = self.var_smoothing * X.var(axis=0).max()
        self.theta_ = np.array([X[y_enc == c].mean(axis=0) for c in range(len(self.classes_))])
        self.var_ = np.array([X[y_enc == c].var(axis=0) for c in range(len(self.classes_))]) + eps
        self.log_prior_ = np.log(np.bincount(y_enc) / len(y_enc))
        return self

    def joint_log_likelihood(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        ll = -0.5 * (
            np.log(2 * np.pi * self.var_)[None, :, :]
            + (X[:, None, :] - self.theta_[None, :, :]) ** 2 / self.var_[None, :, :]
        ).sum(axis=2)
        return ll + self.log_prior_

    def predict_log_proba(self, X) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]


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
