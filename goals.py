from abc import ABC, abstractmethod
from collections import OrderedDict
import re

from convokit import PolitenessStrategies
from detoxify import Detoxify
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
import spacy
from taaled import ld
import textstat
import torch
from tqdm.auto import tqdm
from transformers import pipeline

from beartype import beartype
from typing import Callable, List, Optional, Union

class Goal(ABC):
    @abstractmethod
    def __call__(self, x: str) -> float:
        pass

@beartype
class Goalspace(object):
    """
        This class computes non-normalized goal-space mappings for arbitrary strings. 
    """
    def __init__(
        self,
        goal_dimensions: List[Union[Goal, Callable[str, np.number]]], # the core functionality of a Goal is a mapping from strings to a real number.
    ):
        self.goal_dimensions = goal_dimensions
        self.goal_names = [f.__class__.__name__ for f in self.goal_dimensions]

    def get_goal_names(self, snake_case=False):
        if snake_case:
            return [re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower() for name in self.goal_names]
        else:
            return self.goal_names

    def __call__(
        self,
        x: Union[List[str], str],
        return_pandas: Optional[bool] = True,
    ):
        if not isinstance(x, list):
            x = [x]
        goalspace_mappings = []
        for source_text in tqdm(x):
            mapping = np.array([goal_fn(source_text) for goal_fn in self.goal_dimensions]) # in theory, there should be a more efficient way to parallelize
            goalspace_mappings.append(mapping)
        if return_pandas:
            return pd.DataFrame(goalspace_mappings, columns=self.get_goal_names(snake_case=True))
        else:
            return np.stack(goalspace_mappings, axis=0)

    @beartype
    @classmethod
    def create_default_goalspace_from_probe(
        cls,
        probe: pd.DataFrame,
    ):

        goal_names = [col.split("source_", 1)[1] for col in probe.filter(like="source_", axis=1)]  # goals must be associated with some "source" column
        goal_dimensions = []
        for name in goal_names:
            try:
                default_goal = GoalFactory.get_default(name)
                goal_dimensions.append(default_goal)
            except Exception:
                print("Error when reconstructing goalspace for goal", name)
                continue
        return cls(goal_dimensions)

class GoalFactory:
    _registry = {}

    @classmethod
    def register_goal(cls, key, *default_args, **default_kwargs):
        def decorator(class_type):
            cls._registry[key] = lambda: class_type(*default_args, **default_kwargs)
            return class_type 
        return decorator

    @classmethod
    def get_default(cls, key):
        if key in cls._registry:
            return cls._registry[key]()
        raise ValueError(f"Class {key} is not registered.")

    @classmethod
    def get_registered_goals(cls):
        return cls._registry.keys()

# =============================================================== #
# GOAL UTILITY CLASSES (incl. multi-goal models)                  #
# =============================================================== #

class Model(ABC):
    """
        Utility class. On-demand model loading and calling.
    """
    def __init__(self):
        self.model = None
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load(self, force_reload: Optional[bool] = False):
        pass

    @abstractmethod
    def __call__(self, text: str):
        pass

class SentimentClassifier(Model):
    @beartype
    def load(self, force_reload: Optional[bool] = False):
        if self.model is None or force_reload:
            print("Loading sentiment classifier...")
            self.model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True, device=self.device)
        return self.model

    @beartype
    def __call__(self, text: str):
        sentences = sent_tokenize(text) # this model is trained at the sentence/utterance level, so break it up this way
        sentiment_by_sentences = []
        for sent in sentences: 
            try:
                sent_results = self.model(sent)
                sentiment_by_sentences.append(pd.DataFrame(*sent_results))
            except RuntimeError as e:
                print(f"Sentiment model raised a RuntimeError: {e}. Excluding sentence from sentiment calculation.\nOffending sentence: {sent}")
        if len(sentences) != len(sentiment_by_sentences):
            print(f"Not all sentences were converted successfully ({len(sentiment_by_sentences)}/{len(sentences)}). Consider double-checking this data point.")
        emotion_df = pd.concat(sentiment_by_sentences, keys=range(len(sentiment_by_sentences)))
        return emotion_df.groupby('label').mean()

class DetoxifyModel(Model):
    @beartype
    def load(self, force_reload: Optional[bool] = False):
        if self.model is None or force_reload:
            print("Loading toxicity classifier...")
            self.model = Detoxify('original', device=self.device)
        return self.model

    @beartype
    def __call__(self, text: str):
        sentences = sent_tokenize(text)
        return pd.DataFrame([self.model.predict(sent) for sent in sentences]).mean(axis=0)



class MultiGoalCache(object):
    """
        Utility class to lazy-load models and reduce function calls. This is really just a wrapper
        class around some pre-trained model, backed by an LRU cache. 

	Individual goals for one steerability evaluation should point to the same ModelBasedMultiGoal object, 
        which is equivalent to sharing a model and goal-space mapping cache.
    """
    @beartype
    def __init__(self, model: Model, max_cache_size: Optional[int] = 2048):
        self.max_cache_size = max_cache_size
        self.model = model
        self.cache = OrderedDict()  # Custom cache
 
    def __call__(self, text: str):
        if text in self.cache:
            self.cache.move_to_end(text)
            return self.cache[text]

        self.model.load() 
        result = self.model(text)
        
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False) 
        
        self.cache[text] = result
        return result # the individual goal functions should then post-process this model output

SENTIMENT_CACHE = MultiGoalCache(SentimentClassifier())
DETOXIFY_CACHE = MultiGoalCache(DetoxifyModel())

# =============================================================== #
# GOALS                                                           #
# =============================================================== #

@GoalFactory.register_goal("reading_difficulty")
class ReadingDifficulty(Goal):

    @beartype
    def __call__(self, text: str):
        return textstat.flesch_kincaid_grade(text)


@GoalFactory.register_goal("politeness")
class Politeness(Goal):
    _spacy_model = None  
    _politeness_model = None
    _politeness_weights = pd.DataFrame([{ # copied from Table 3, "A computational approach to politeness" (Danescu-Niculescu-Mizil 2012)
        'feature_politeness_==Please==': 0.49,
        'feature_politeness_==Please_start==': -0.30,
        'feature_politeness_==HASHEDGE==': 0.0, # too general -- not in Table 3
        'feature_politeness_==Indirect_(btw)==': 0.63,
        'feature_politeness_==Hedges==': 0.14,
        'feature_politeness_==Factuality==': -0.38,
        'feature_politeness_==Deference==': 0.78,
        'feature_politeness_==Gratitude==': 0.87,
        'feature_politeness_==Apologizing==': 0.36,
        'feature_politeness_==1st_person_pl.==': 0.08,
        'feature_politeness_==1st_person==': 0.08,
        'feature_politeness_==1st_person_start==': 0.12,
        'feature_politeness_==2nd_person==': 0.05,
        'feature_politeness_==2nd_person_start==': -0.30,
        'feature_politeness_==Indirect_(greeting)==': 0.43,
        'feature_politeness_==Direct_question==': -0.27,
        'feature_politeness_==Direct_start==': -0.43,
        'feature_politeness_==HASPOSITIVE==': 0.12,
        'feature_politeness_==HASNEGATIVE==': -0.13,
        'feature_politeness_==SUBJUNCTIVE==': 0.47,
        'feature_politeness_==INDICATIVE==': 0.09,
    }])

    @classmethod
    def get_politeness_model(cls):
        if cls._politeness_model is None:
            cls._politeness_model = PolitenessStrategies(verbose=1000)
        return cls._politeness_model

    @classmethod
    def get_spacy_model(cls):
        if cls._spacy_model is None:
            cls._spacy_model = spacy.load("en_core_web_sm")
        return cls._spacy_model 

    @beartype
    def __call__(self, text: str):
        sentences = sent_tokenize(text)
        politeness_model = self.get_politeness_model()
        politeness_df = pd.DataFrame([politeness_model.transform_utterance(sent, spacy_nlp=self.get_spacy_model()).meta["politeness_strategies"] for sent in sentences])        
        return (politeness_df.mean(axis=0) @ self._politeness_weights.T).item()

class HFSentiment(Goal):
    def __init__(self, goal_model_cache: MultiGoalCache, sentiment_name: str):
        self.goal_model_cache = goal_model_cache
        self.sentiment_name = sentiment_name

    @beartype
    def __call__(self, text: str):
        return self.goal_model_cache(text).loc[self.sentiment_name].item() # D.R.Y.


@GoalFactory.register_goal("anger", goal_model_cache=SENTIMENT_CACHE)
class Anger(HFSentiment):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "anger")

@GoalFactory.register_goal("disgust", goal_model_cache=SENTIMENT_CACHE)
class Disgust(HFSentiment):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "disgust")

@GoalFactory.register_goal("fear", goal_model_cache=SENTIMENT_CACHE)
class Fear(HFSentiment):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "fear")

@GoalFactory.register_goal("joy", goal_model_cache=SENTIMENT_CACHE)
class Joy(HFSentiment):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "joy")

@GoalFactory.register_goal("sadness", goal_model_cache=SENTIMENT_CACHE)
class Sadness(HFSentiment):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "sadness")

@GoalFactory.register_goal("surprise", goal_model_cache=SENTIMENT_CACHE)
class Surprise(HFSentiment):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "surprise")

@GoalFactory.register_goal("textual_diversity")
class TextualDiversity(Goal):
    def __call__(self, text: str):
        if len(word_tokenize(text)) < 50:
            print("Less than 50 words in evaluation text. This may lead to an inaccurate diversity measure.")
        ldvals = ld.lexdiv(text)
        return ldvals.mtld 

@GoalFactory.register_goal("text_length")
class TextLength(Goal):
    @beartype
    def __call__(self, text: str):
        return len(word_tokenize(text))

class DetoxifyAspect(Goal):
    def __init__(self, goal_model_cache: MultiGoalCache, aspect_name: str):
        self.goal_model_cache = goal_model_cache
        self.aspect_name = aspect_name

    @beartype
    def __call__(self, text: str):
        return self.goal_model_cache(text).loc[self.aspect_name]


@GoalFactory.register_goal("toxicity", goal_model_cache=DETOXIFY_CACHE)
class Toxicity(DetoxifyAspect):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "toxicity")

@GoalFactory.register_goal("severe_toxicity", goal_model_cache=DETOXIFY_CACHE)
class SevereToxicity(DetoxifyAspect):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "severe_toxicity")

@GoalFactory.register_goal("obscene", goal_model_cache=DETOXIFY_CACHE)
class Obscene(DetoxifyAspect):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "obscene")

@GoalFactory.register_goal("threat", goal_model_cache=DETOXIFY_CACHE)
class Threat(DetoxifyAspect):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "threat")

@GoalFactory.register_goal("insult", goal_model_cache=DETOXIFY_CACHE)
class Insult(DetoxifyAspect):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "insult")

@GoalFactory.register_goal("identity_attack", goal_model_cache=DETOXIFY_CACHE)
class IdentityAttack(DetoxifyAspect):
    def __init__(self, goal_model_cache: MultiGoalCache):
        super().__init__(goal_model_cache, "identity_attack")

DEFAULT_GOALS = [
    GoalFactory.get_default("reading_difficulty"),
    GoalFactory.get_default("politeness"),
    GoalFactory.get_default("anger"),
    GoalFactory.get_default("disgust"),
    GoalFactory.get_default("fear"),
    GoalFactory.get_default("joy"),
    GoalFactory.get_default("sadness"),
    GoalFactory.get_default("surprise"),
    GoalFactory.get_default("textual_diversity"),
    GoalFactory.get_default("text_length")
]
TOXICITY_GOALS = [
    GoalFactory.get_default("toxicity"),
    GoalFactory.get_default("severe_toxicity"),
    GoalFactory.get_default("obscene"), 
    GoalFactory.get_default("threat"),
    GoalFactory.get_default("insult"),
    GoalFactory.get_default("identity_attack"),
]
ALL_GOALS = DEFAULT_GOALS + TOXICITY_GOALS

