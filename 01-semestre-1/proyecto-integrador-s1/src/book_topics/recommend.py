"""Temas → libros: perfil de temas del lector, puntaje por temas, híbrido y explicaciones."""

import numpy as np
import pandas as pd
from scipy import sparse


def center_ratings(df: pd.DataFrame, n_users: int, n_books: int) -> sparse.csr_matrix:
    """Calificación − media del lector: un 3 de quien da 5 a todo es una señal negativa."""
    centered = df["rating"] - df.groupby("u")["rating"].transform("mean")
    return sparse.csr_matrix(
        (centered.to_numpy(float), (df["u"], df["i"])), shape=(n_users, n_books)
    )


def user_topic_profiles(R: sparse.csr_matrix, T: np.ndarray) -> np.ndarray:
    return np.asarray(R @ T)


def profile_from_titles(liked: list[int], disliked: list[int], T: np.ndarray) -> np.ndarray:
    out = np.zeros(T.shape[1])
    if liked:
        out += T[liked].sum(axis=0)
    if disliked:
        out -= T[disliked].sum(axis=0)
    return out


def topic_scores(profiles: np.ndarray, T: np.ndarray) -> np.ndarray:
    pn = np.linalg.norm(profiles, axis=1, keepdims=True)
    tn = np.linalg.norm(T, axis=1, keepdims=True)
    P = np.divide(profiles, pn, out=np.zeros_like(profiles, dtype=float), where=pn > 0)
    Tn = np.divide(T, tn, out=np.zeros_like(T, dtype=float), where=tn > 0)
    return P @ Tn.T


def zscore_rows(S: np.ndarray) -> np.ndarray:
    mu, sd = S.mean(axis=1, keepdims=True), S.std(axis=1, keepdims=True)
    return np.divide(S - mu, sd, out=np.zeros_like(S, dtype=float), where=sd > 0)


def hybrid_scores(collab: np.ndarray, topic: np.ndarray, alpha: float) -> np.ndarray:
    """α·z(colaborativo) + (1 − α)·z(temas): estandarizar por lector iguala las escalas."""
    return alpha * zscore_rows(collab) + (1 - alpha) * zscore_rows(topic)


def explain(
    profile: np.ndarray, topic_vec: np.ndarray, names: list[str], top: int = 3
) -> list[str]:
    contrib = profile * topic_vec
    return [names[t] for t in np.argsort(-contrib)[:top] if contrib[t] > 0]
