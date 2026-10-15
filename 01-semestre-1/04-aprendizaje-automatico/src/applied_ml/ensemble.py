"""Ensambles sobre los árboles propios: random forest (bagging) y gradient boosting."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from applied_ml.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(ClassifierMixin, BaseEstimator):
    """Bagging de árboles profundos + submuestreo de features en cada nodo.

    Promediar B árboles con varianza σ² y correlación ρ da varianza ρσ² + (1−ρ)σ²/B:
    el bootstrap reduce el segundo término; `max_features` baja ρ, que es lo que limita.
    Las muestras fuera de la bolsa (~36.8 % por árbol) dan una estimación de error gratis (OOB).
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="sqrt",
        max_depth=None,
        min_samples_leaf=1,
        bootstrap=True,
        oob_score=False,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.estimators_ = []
        oob_sum = np.zeros((n, len(self.classes_)))
        oob_count = np.zeros(n)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(2**31)),
            )
            tree.fit(X[idx], y_enc[idx])
            # el árbol puede no haber visto todas las clases: alinear columnas
            proba = np.zeros((n, len(self.classes_)))
            self.estimators_.append(tree)
            if self.oob_score and self.bootstrap:
                oob = np.setdiff1d(np.arange(n), idx)
                proba_oob = tree.predict_proba(X[oob])
                proba[oob[:, None], tree.classes_[None, :]] = proba_oob
                oob_sum[oob] += proba[oob]
                oob_count[oob] += 1
        if self.oob_score and self.bootstrap:
            seen = oob_count > 0
            self.oob_score_ = float(np.mean(oob_sum[seen].argmax(axis=1) == y_enc[seen]))
        imps = np.array([t.feature_importances_ for t in self.estimators_])
        self.feature_importances_ = imps.mean(axis=0)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.zeros((len(X), len(self.classes_)))
        for tree in self.estimators_:
            out[:, tree.classes_] += tree.predict_proba(X)
        return out / len(self.estimators_)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class GradientBoostingClassifier(ClassifierMixin, BaseEstimator):
    """Boosting binario con pérdida logística: suma de árboles de regresión poco profundos.

    En cada ronda se ajusta un árbol al gradiente negativo r = y − p y luego, en cada hoja,
    se reemplaza su valor por un paso de Newton: Σr / Σp(1−p). `learning_rate` encoge cada
    paso (shrinkage): más rondas, mejor generalización. `subsample` < 1 da boosting estocástico.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("Solo clasificación binaria")
        y_enc = y_enc.astype(float)
        rng = np.random.default_rng(self.random_state)
        p0 = np.clip(y_enc.mean(), 1e-12, 1 - 1e-12)
        self.init_ = float(np.log(p0 / (1 - p0)))  # log-odds de la prevalencia
        f = np.full(len(X), self.init_)
        self.estimators_, self.leaf_values_ = [], []
        self.train_loss_ = []
        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-f))
            r = y_enc - p
            idx = (
                rng.choice(len(X), int(self.subsample * len(X)), replace=False)
                if self.subsample < 1
                else np.arange(len(X))
            )
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X[idx], r[idx])
            leaves = tree.apply(X[idx])
            values = np.zeros(len(tree.tree_feature_))
            num = np.bincount(leaves, weights=r[idx], minlength=len(values))
            den = np.bincount(leaves, weights=(p * (1 - p))[idx], minlength=len(values))
            np.divide(num, den, out=values, where=den > 1e-12)
            self.estimators_.append(tree)
            self.leaf_values_.append(values)
            f += self.learning_rate * values[tree.apply(X)]
            self.train_loss_.append(float(np.mean(np.logaddexp(0, f) - y_enc * f)))
        return self

    def staged_decision_function(self, X):
        X = np.asarray(X, dtype=float)
        f = np.full(len(X), self.init_)
        for tree, values in zip(self.estimators_, self.leaf_values_, strict=True):
            f = f + self.learning_rate * values[tree.apply(X)]
            yield f

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        f = np.full(len(X), self.init_)
        for tree, values in zip(self.estimators_, self.leaf_values_, strict=True):
            f += self.learning_rate * values[tree.apply(X)]
        return f

    def predict_proba(self, X) -> np.ndarray:
        p1 = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
