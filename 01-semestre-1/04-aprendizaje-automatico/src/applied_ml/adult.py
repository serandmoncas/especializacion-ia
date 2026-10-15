"""Carga y limpieza de UCI Adult. Reutilizable por otros módulos (p. ej. ética)."""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
TARGET = "income"
# Atributos protegidos: se conservan para auditar el modelo, no para justificar su uso como feature.
PROTECTED = ["sex", "race"]


def _read(path: Path, *, skip_first: bool) -> pd.DataFrame:
    df = pd.read_csv(
        path, names=COLUMNS, skipinitialspace=True, na_values="?", skiprows=1 if skip_first else 0
    )
    df = df.dropna(subset=[TARGET])  # adult.test termina con una línea vacía
    # adult.test escribe las etiquetas con punto final (">50K.")
    df[TARGET] = df[TARGET].str.rstrip(".").eq(">50K").astype(int)
    return df


def load_adult(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) con las particiones oficiales. Objetivo `income` ∈ {0, 1} (1 = >50K).

    Limpieza:
    - `fnlwgt` se elimina: es el peso muestral del censo (cuántas personas representa la fila),
      no un atributo de la persona.
    - `education` se elimina: es una recodificación 1 a 1 de `education_num`.
    - Los faltantes ("?") quedan como NaN; se imputan dentro del Pipeline, no aquí, para que
      la imputación se aprenda solo con datos de entrenamiento.
    """
    data_dir = Path(data_dir)
    train = _read(data_dir / "adult.data", skip_first=False)
    test = _read(data_dir / "adult.test", skip_first=True)  # primera línea: "|1x3 Cross validator"
    drop = ["fnlwgt", "education"]
    return train.drop(columns=drop).reset_index(drop=True), test.drop(columns=drop).reset_index(
        drop=True
    )
