import numpy as np
import pandas as pd
from book_topics.classify import (
    KNNTopics,
    PrevalenceModel,
    TextTopicModel,
    apply_thresholds,
    book_text,
    fit_thresholds,
)

TEXTS = (
    ["dragon magic quest fire"] * 20
    + ["love romance paris kiss"] * 20
    + ["detective murder crime clue"] * 20
)
Y = np.zeros((60, 3), bool)
Y[:20, 0], Y[20:40, 1], Y[40:, 2] = True, True, True


def test_book_text_joins_title_and_description():
    assert book_text(pd.Series(["T"], index=[1]), pd.Series(["Desc."], index=[1]))[1] == "T. Desc."


def test_prevalence_model():
    p = PrevalenceModel().fit(Y).predict_proba(2)
    assert p.shape == (2, 3) and np.allclose(p, 1 / 3)


def test_text_model_learns_obvious_topics():
    p = (
        TextTopicModel(min_df=1)
        .fit(TEXTS, Y)
        .predict_proba(["a dragon and magic", "a murder clue"])
    )
    assert p[0].argmax() == 0 and p[1].argmax() == 2


def test_knn_votes_by_similarity():
    model = TextTopicModel(min_df=1).fit(TEXTS, Y)
    knn = KNNTopics(k=5).fit(model.transform(TEXTS), Y)
    assert knn.predict_proba(model.transform(["love in paris"]))[0].argmax() == 1


def test_thresholds_and_at_least_one_topic():
    P = np.array([[0.2, 0.1], [0.9, 0.8]])
    pred = apply_thresholds(P, np.array([0.5, 0.5]))
    assert pred[0].tolist() == [True, False]  # ninguno supera: el más cercano a su umbral
    assert pred[1].all()
    assert fit_thresholds(np.array([[1, 0], [1, 1]], bool), P).shape == (2,)


def test_out_of_vocabulary_text_is_low_confidence():
    model = TextTopicModel(min_df=1).fit(TEXTS, Y)
    assert model.low_confidence(["asdf qwerty zzzz", "dragon magic"]).tolist() == [True, False]
    assert np.allclose(model.predict_proba(["asdf qwerty zzzz"])[0], Y.mean(axis=0))
