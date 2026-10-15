"""Aprendizaje automático desde cero, con API compatible con scikit-learn.

Los estimadores heredan de `BaseEstimator`: el núcleo es propio, pero funcionan dentro de
`Pipeline`, `cross_val_score` y `GridSearchCV`, lo que permite compararlos con la librería
en igualdad de condiciones.
"""

from applied_ml.cluster import KMeans
from applied_ml.ensemble import GradientBoostingClassifier, RandomForestClassifier
from applied_ml.linear import LogisticRegression
from applied_ml.tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GradientBoostingClassifier",
    "KMeans",
    "LogisticRegression",
    "RandomForestClassifier",
]
