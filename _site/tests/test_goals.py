import pytest

import numpy as np
import pandas as pd

from goals import (
    ReadingDifficulty,
    Politeness,
    Anger,
    Disgust,
    Fear,
    Joy,
    Sadness,
    Surprise,
    TextualDiversity,
    TextLength,
    Toxicity,
    SevereToxicity,
    Insult,
    Threat,
    Obscene,
    IdentityAttack,
    SentimentClassifier,
    DetoxifyModel,
    MultiGoalCache, 
    Goalspace,
    )


def is_numeric(value):
    return isinstance(value, (int, float)) or np.issubdtype(type(value), np.number)

@pytest.fixture
def goalspace():
    return Goalspace(goal_dimensions=[
        ReadingDifficulty(),
        Politeness(),
        TextualDiversity(),
        TextLength(),
        Anger(MultiGoalCache(SentimentClassifier())),
        Disgust(MultiGoalCache(SentimentClassifier())),
        Fear(MultiGoalCache(SentimentClassifier())),
        Joy(MultiGoalCache(SentimentClassifier())),
        Sadness(MultiGoalCache(SentimentClassifier())),
        Surprise(MultiGoalCache(SentimentClassifier())),
        Toxicity(MultiGoalCache(DetoxifyModel())),
        SevereToxicity(MultiGoalCache(DetoxifyModel())),
        Insult(MultiGoalCache(DetoxifyModel())),
        Threat(MultiGoalCache(DetoxifyModel())),
        Obscene(MultiGoalCache(DetoxifyModel())),
        IdentityAttack(MultiGoalCache(DetoxifyModel())),
        ])

@pytest.fixture
def source_texts():
    return ["Hello! I like cats!", "Dogs are fun. Like and subscribe if you agree.", "Foxes are simply dog hardware with cat software."]

@pytest.mark.parametrize("goal_class", [ReadingDifficulty, Politeness, TextualDiversity, TextLength])
def test_goal_returns_int_or_float(goal_class, source_texts):
    goal = goal_class()
    assert is_numeric(goal(source_texts[0]))

@pytest.mark.parametrize("sentiment_class", [Anger, Disgust, Fear, Joy, Sadness, Surprise])
def test_sentiment_returns_int_or_float(sentiment_class, source_texts):
    goal = sentiment_class(MultiGoalCache(SentimentClassifier()))
    assert is_numeric(goal(source_texts[0]))

@pytest.mark.parametrize("toxicity_aspect", [Toxicity, SevereToxicity, Insult, Threat, Obscene, IdentityAttack])
def test_toxicity_returns_int_or_float(toxicity_aspect, source_texts):
    goal = toxicity_aspect(MultiGoalCache(DetoxifyModel()))
    assert is_numeric(goal(source_texts[0]))

def test_goalspace_numpy(goalspace, source_texts):
    gsm = goalspace(source_texts, return_pandas=False)
    assert isinstance(gsm, np.ndarray)
    assert gsm.shape == (len(source_texts), len(goalspace.goal_dimensions))

def test_goalspace_pandas(goalspace, source_texts):
    gsm = goalspace(source_texts, return_pandas=True)
    assert isinstance(gsm, pd.DataFrame)
    assert gsm.shape == (len(source_texts), len(goalspace.goal_dimensions))
    assert gsm.columns.tolist() == [
            "ReadingDifficulty", "Politeness", "TextualDiversity", "TextLength",
            "Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise",
            "Toxicity", "SevereToxicity", "Insult", "Threat", "Obscene", "IdentityAttack"
        ]
