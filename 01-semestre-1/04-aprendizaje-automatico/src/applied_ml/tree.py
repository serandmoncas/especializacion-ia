"""Árboles de decisión CART para clasificación (Gini/entropía) y regresión (MSE).

La búsqueda del mejor corte es vectorizada: por cada feature se ordenan las muestras una vez
y las impurezas de *todos* los cortes candidatos salen de sumas acumuladas, en O(n log n)
por feature y nodo en lugar de O(n²).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

LEAF = -1


def _gini(counts: np.ndarray) -> np.ndarray:
    """Gini de cada fila de conteos [..., k]: 1 − Σ p²."""
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    return 1 - (p**2).sum(axis=-1)


def _entropy(counts: np.ndarray) -> np.ndarray:
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=-1)


class _TreeBuilder:
    """Construye el árbol en arreglos planos: feature, threshold, hijos, valor, n, impureza."""

    def __init__(
        self, task, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, rng
    ):
        self.task, self.criterion = task, criterion
        self.max_depth = np.inf if max_depth is None else max_depth
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.max_features, self.rng = max_features, rng
        self.feature, self.threshold, self.left, self.right = [], [], [], []
        self.value, self.n, self.impurity = [], [], []

    # -- impureza y valor de hoja --------------------------------------------------------------
    def _node_stats(self, y: np.ndarray) -> tuple[np.ndarray, float]:
        if self.task == "clf":
            counts = y.sum(axis=0)  # y viene one-hot [n, k]
            imp = _gini(counts) if self.criterion == "gini" else _entropy(counts)
            return counts / counts.sum(), float(imp)
        return np.array([y.mean()]), float(y.var())

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float, float] | None:
        n, d = X.shape
        msl = self.min_samples_leaf
        if self.max_features is None:
            feats = np.arange(d)
        else:
            feats = self.rng.choice(d, size=min(self.max_features, d), replace=False)
        best = None  # (impureza ponderada, feature, threshold)
        for j in feats:
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            # candidatos: posiciones i donde xs[i-1] < xs[i] y ambos lados tienen ≥ msl muestras
            pos = np.arange(msl, n - msl + 1)
            pos = pos[xs[pos - 1] < xs[pos]] if len(pos) else pos
            if len(pos) == 0:
                continue
            if self.task == "clf":
                cum = np.cumsum(y[order], axis=0)  # [n, k]
                left = cum[pos - 1]
                right = cum[-1] - left
                f = _gini if self.criterion == "gini" else _entropy
                score = (pos * f(left) + (n - pos) * f(right)) / n
            else:
                ys = y[order]
                c1, c2 = np.cumsum(ys), np.cumsum(ys**2)
                nl, nr = pos, n - pos
                sl, sl2 = c1[pos - 1], c2[pos - 1]
                sr, sr2 = c1[-1] - sl, c2[-1] - sl2
                # n·var = Σy² − (Σy)²/n en cada lado
                score = ((sl2 - sl**2 / nl) + (sr2 - sr**2 / nr)) / n
            k = int(np.argmin(score))
            if best is None or score[k] < best[0] - 1e-12:
                i = pos[k]
                best = (float(score[k]), int(j), float((xs[i - 1] + xs[i]) / 2))
        return None if best is None else (best[1], best[2], best[0])

    def build(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> int:
        node = len(self.feature)
        value, imp = self._node_stats(y)
        for lst, v in (
            (self.feature, LEAF),
            (self.threshold, np.nan),
            (self.left, LEAF),
            (self.right, LEAF),
            (self.value, value),
            (self.n, len(y)),
            (self.impurity, imp),
        ):
            lst.append(v)
        if depth >= self.max_depth or len(y) < self.min_samples_split or imp <= 1e-12:
            return node
        split = self._best_split(X, y)
        # Se aceptan cortes con ganancia cero (como sklearn con min_impurity_decrease=0): en XOR
        # ningún corte de la raíz mejora la impureza, pero sin él el nivel siguiente no existe.
        # Exigir mejora estricta vuelve al árbol miope.
        if split is None or split[2] > imp + 1e-12:
            return node
        j, thr, _ = split
        mask = X[:, j] <= thr
        self.feature[node], self.threshold[node] = j, thr
        self.left[node] = self.build(X[mask], y[mask], depth + 1)
        self.right[node] = self.build(X[~mask], y[~mask], depth + 1)
        return node


class _BaseTree(BaseEstimator):
    _task = ""

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _resolve_max_features(self, d: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return int(mf)

    def _fit(self, X, y_target):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        b = _TreeBuilder(
            self._task,
            self.criterion,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            self._resolve_max_features(X.shape[1]),
            np.random.default_rng(self.random_state),
        )
        b.build(X, y_target)
        self.tree_feature_ = np.array(b.feature)
        self.tree_threshold_ = np.array(b.threshold)
        self.tree_left_ = np.array(b.left)
        self.tree_right_ = np.array(b.right)
        self.tree_value_ = np.array(b.value)
        self.tree_n_ = np.array(b.n)
        self.tree_impurity_ = np.array(b.impurity)
        self._compute_importances()
        return self

    def _compute_importances(self) -> None:
        """Importancia = disminución total de impureza ponderada por muestras, normalizada."""
        imp = np.zeros(self.n_features_in_)
        for node in np.flatnonzero(self.tree_feature_ != LEAF):
            left, right = self.tree_left_[node], self.tree_right_[node]
            decrease = (
                self.tree_n_[node] * self.tree_impurity_[node]
                - self.tree_n_[left] * self.tree_impurity_[left]
                - self.tree_n_[right] * self.tree_impurity_[right]
            )
            imp[self.tree_feature_[node]] += decrease
        total = imp.sum()
        self.feature_importances_ = imp / total if total > 0 else imp

    def apply(self, X) -> np.ndarray:
        """Hoja a la que cae cada muestra, particionando índices (sin bucle por muestra)."""
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X), dtype=int)
        stack = [(0, np.arange(len(X)))]
        while stack:
            node, idx = stack.pop()
            if self.tree_feature_[node] == LEAF or len(idx) == 0:
                out[idx] = node
                continue
            go_left = X[idx, self.tree_feature_[node]] <= self.tree_threshold_[node]
            stack.append((self.tree_left_[node], idx[go_left]))
            stack.append((self.tree_right_[node], idx[~go_left]))
        return out

    @property
    def n_leaves_(self) -> int:
        return int(np.sum(self.tree_feature_ == LEAF))

    def get_depth(self) -> int:
        depth = np.zeros(len(self.tree_feature_), dtype=int)
        for node in range(len(self.tree_feature_)):  # los hijos siempre tienen índice mayor
            if self.tree_feature_[node] != LEAF:
                depth[self.tree_left_[node]] = depth[self.tree_right_[node]] = depth[node] + 1
        return int(depth.max())


class DecisionTreeClassifier(ClassifierMixin, _BaseTree):
    _task = "clf"

    def fit(self, X, y):
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        onehot = np.eye(len(self.classes_))[y_enc]
        return self._fit(X, onehot)

    def predict_proba(self, X) -> np.ndarray:
        return self.tree_value_[self.apply(X)]

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class DecisionTreeRegressor(RegressorMixin, _BaseTree):
    _task = "reg"

    def __init__(
        self,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        super().__init__(
            criterion, max_depth, min_samples_split, min_samples_leaf, max_features, random_state
        )

    def fit(self, X, y):
        return self._fit(X, np.asarray(y, dtype=float))

    def predict(self, X) -> np.ndarray:
        return self.tree_value_[self.apply(X), 0]
