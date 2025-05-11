from abc import ABC, abstractmethod
import atexit
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

from filelock import FileLock
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from pylats import lats
import spacy
from taaled import ld
import textstat
from tqdm.auto import tqdm

from beartype import beartype
from beartype.typing import Callable, List
from typing import Optional, Union

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
        goal_dimensions: List[Union[Goal, Callable[[str], np.number]]], # the core functionality of a Goal is a mapping from strings to a real number.
        cache_path: Optional[str] = "cache/goalspace.json",
    ):
        self.goal_dimensions = goal_dimensions
        self.goal_names = [f.__class__.__name__ for f in self.goal_dimensions]
        self.cache_path = cache_path
        self.cache_modified = False
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r") as f:
                    self.cache = json.load(f)
            except Exception:
                print("Failed to load cache:", cache_path)
                self.cache = {}
            atexit.register(self.save_cache)
        elif self.cache_path is None:
            self.cache = {}
        else:
            self.cache = {}
            atexit.register(self.save_cache)

    def save_cache(self):
        if self.cache_path is None:
            print("Cache path is None. Saving skipped.")
            return
        if not self.cache_modified:
            print("Cache has not been modified.")
            return

        print("Saving goalspace-mapping cache...")
        # sync w/ copy on disk. note that this is NOT threadsafe
        curr_cache = self.cache
        lock = FileLock(self.cache_path + ".lock")
        with lock:
            if os.path.isfile(self.cache_path):
                with open(self.cache_path, "r") as f:
                    cache_on_disk = json.load(f)
                curr_cache = {**cache_on_disk, **self.cache} # self.cache will overwrite 
            with open(self.cache_path, 'w') as f:
                json.dump(curr_cache, f)

    def get_goal_names(self, snake_case=False):
        if snake_case:
            return [re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower() for name in self.goal_names]
        else:
            return self.goal_names

    def encode_key(self, key: str):
        encoded_key = base64.urlsafe_b64encode(key.encode()).decode()
        return encoded_key

    def __call__(
        self,
        texts: Union[List[str], str],
        return_pandas: Optional[bool] = True,
        show_progress: Optional[bool] = True,
        max_workers: Optional[int] = 1,
    ):
        if not isinstance(texts, list):
            texts = [texts]
        goalspace_mappings = []

        def process_single_text(source_text):
            key = self.encode_key(source_text)
            try:
                raw_mapping = self.cache[key]
                return np.array([raw_mapping[goal_fn.__class__.__name__] for goal_fn in self.goal_dimensions])
            except KeyError:
                mapping = []
                for goal_fn in self.goal_dimensions:
                    try:
                        goal_val = goal_fn(source_text)
                    except RuntimeError as e:
                        problem_goal = goal_fn.__class__.__name__
                        print(f"MAPPING FAILED DURING GOAL - {problem_goal}")
                        print("Failed to map source text:", source_text)
                        problem_file = f".problem_text_for_{problem_goal}"
                        with open(problem_file, "w") as f:
                            f.write(source_text)
                        print("Saved problem text to:", problem_file)
                        raise e
                    mapping.append(goal_val)
                mapping = np.array(mapping)

                curr_raw_mapping = self.cache.get(source_text, {})
                mapping_serialized = {
                    map_fn.__class__.__name__: map_val for map_fn, map_val in zip(self.goal_dimensions, mapping)
                }

                if self.cache_path is not None:
                    self.cache[key] = {**curr_raw_mapping, **mapping_serialized}
                    self.cache_modified = True

                return mapping

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                goalspace_mappings = list(tqdm(executor.map(process_single_text, texts), total=len(texts), disable=not show_progress))
        else:
            goalspace_mappings = [process_single_text(text) for text in tqdm(texts, disable=not show_progress)]

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
        print("Creating goalspace with dimensions:", goal_dimensions)
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
# GOALS                                                           #
# =============================================================== #

@GoalFactory.register_goal("reading_difficulty")
class ReadingDifficulty(Goal):

    @beartype
    def __call__(self, text: str):
        return textstat.flesch_kincaid_grade(text)

@GoalFactory.register_goal("textual_diversity")
class TextualDiversity(Goal):
    def __call__(self, text: str):
        if len(word_tokenize(text)) < 50:
            print("Less than 50 words in evaluation text. This may lead to an inaccurate diversity measure.")
        cleaned = lats.Normalize(text, lats.ld_params_en)
        if len(cleaned.toks) == 0 or len(set(cleaned.toks)) == 0: # this will cause some calculation issues, so we assume that such texts are trivially non-diverse
            return 0.
        ldvals = ld.lexdiv(cleaned.toks)
        return ldvals.mtld 

@GoalFactory.register_goal("text_length")
class TextLength(Goal):
    @beartype
    def __call__(self, text: str):
        return len(word_tokenize(text))
    
@GoalFactory.register_goal("formality")
class Formality(Goal):
    # via the Heylighen-Dewaele score 
    _spacy_model = None

    @classmethod
    def get_spacy_model(cls):
        if cls._spacy_model is None:
            cls._spacy_model = spacy.load("en_core_web_sm")
        return cls._spacy_model 
    
    @beartype
    def get_pos_freqs(self, text: str):
        spacy_model = self.get_spacy_model()
        pos_counts = {
            "NOUN": 0,
            "ADJ": 0,
            "ADP": 0,  
            "ARTICLE": 0, # IMPORTANT! We cannot use "DET" directly because the/a/an are non-diectic, but this/that are! See Kleiber (1991), 
            "PRON": 0,
            "VERB": 0,
            "ADV": 0,
            "INTJ": 0
        }
        doc = spacy_model(text)
        total = 0
        for token in doc:
            if token.is_alpha:
                total += 1
                pos = token.pos_
                word = token.text.lower()
                if pos in pos_counts and pos != "DET":
                    pos_counts[pos] += 1
                elif pos == "DET" and word in {"a", "an", "the"}:
                    pos_counts["ARTICLE"] += 1
        freqs = {k: 0 if total == 0 else v / total * 100 for k, v in pos_counts.items()}
        return freqs

    @beartype
    def __call__(self, text: str):
        freqs = self.get_pos_freqs(text)
        # Apply Heylighen & Dewaele formula (pg. 309, VARIATION IN THE CONTEXTUALITY OF LANGUAGE)
        diectic_coeff = freqs["NOUN"] + freqs["ADJ"] + freqs["ADP"] + freqs["ARTICLE"]
        non_diectic_coeff = freqs["PRON"] + freqs["VERB"] + freqs["ADV"] + freqs["INTJ"]
        F = (diectic_coeff - non_diectic_coeff + 100) / 2
        return F


DEFAULT_GOALS = [
    GoalFactory.get_default("reading_difficulty"),
    GoalFactory.get_default("textual_diversity"),
    GoalFactory.get_default("text_length"),
    GoalFactory.get_default("formality"),
]
ALL_GOALS = DEFAULT_GOALS # in case, in the future we want to add more types of goals (e.g., STYLE_GOALS, UNSAFE_GOALS)

