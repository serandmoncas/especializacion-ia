# ruff: noqa: E501  (líneas de datos crudos del formato Adult)
from applied_ml.adult import CATEGORICAL, NUMERIC, TARGET, load_adult

TRAIN = """39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, ?, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, >50K
"""
TEST = """|1x3 Cross validator
25, Private, 226802, 11th, 7, Never-married, ?, Own-child, Black, Male, 0, 0, 40, United-States, <=50K.
38, Private, 89814, HS-grad, 9, Married-civ-spouse, Farming-fishing, Husband, White, Female, 0, 0, 50, ?, >50K.

"""


def test_load_adult_handles_file_quirks(tmp_path):
    (tmp_path / "adult.data").write_text(TRAIN)
    (tmp_path / "adult.test").write_text(TEST)
    train, test = load_adult(tmp_path)
    assert len(train) == 2 and len(test) == 2
    assert train[TARGET].tolist() == [0, 1]
    assert test[TARGET].tolist() == [0, 1]  # el punto final de adult.test no rompe las etiquetas
    assert train["workclass"].isna().sum() == 1  # "?" → NaN, sin espacios residuales
    assert test["native_country"].isna().sum() == 1
    assert "fnlwgt" not in train and "education" not in train
    assert set(NUMERIC + CATEGORICAL + [TARGET]) == set(train.columns)
    assert test.loc[1, "sex"] == "Female"
