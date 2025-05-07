from abc import ABC, abstractmethod
import atexit
import base64
import bisect
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re

try:
    from convokit import PolitenessStrategies
except ModuleNotFoundError as e:
    print("Unable to import convokit:", e)

try:
    from detoxify import Detoxify
except ModuleNotFoundError as e:
    print("Unable to import detoxify:", e)

try:
    from empath import Empath
except ImportError:
    print("Unable to import empath.")

from filelock import FileLock
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from pylats import lats
import spacy
from taaled import ld
import textstat
import torch
from tqdm.auto import tqdm

try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
except RuntimeError as e:
    print("Failed to import transformers -- check torch version and torchvision compatibility", e)
import warnings

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
# GOAL UTILITY CLASSES (incl. multi-goal models)                  #
# =============================================================== #

class Model(ABC):
    """
        Utility class. On-demand model loading and calling.
    """
    def __init__(self, device: Optional[str] = None):
        self.model = None
        if device is None:
            num_gpus = torch.cuda.device_count()
            self.device = f"cuda:{num_gpus-1}" if torch.cuda.is_available() else "cpu" # ....we'll deal with this later
        else:
            self.device = device
        warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

    @abstractmethod
    def load(self, force_reload: Optional[bool] = False):
        pass

    @abstractmethod
    def __call__(self, text: str):
        pass

    @beartype
    def _rechunk(self, sent: str):
        token_counts = []
        words = sent.split()
        for word in words:
            if word in self.tokenizer_cache:
                input_ids = self.tokenizer_cache[word]
            else:
                input_ids = self.model.tokenizer(word)["input_ids"]
                self.tokenizer_cache[word] = input_ids
            token_counts.append(len(input_ids))
        
        cumulative_token_counts = np.cumsum(token_counts)
        limits = list(range(self.max_len, np.sum(token_counts) + self.max_len, self.max_len))

        start_idx = 0
        chunks = []
        
        for limit in limits:
            idx = bisect.bisect_right(cumulative_token_counts, limit)  # No need for `-1`
            if idx <= start_idx:  # Ensure at least one word is included
                idx = min(start_idx + 1, len(words))
            chunk = " ".join(words[start_idx:idx])
            chunks.append(chunk)
            start_idx = idx  # Move start index forward

            if start_idx >= len(words):  # Stop if all words are covered
                break

        return chunks


class SentimentClassifier(Model):
    CHINESE_PATTERN = re.compile(r"[\u4E00-\u9FFF]")

    @beartype
    def load(self, force_reload: Optional[bool] = False):
        if self.model is None or force_reload:
            print(f"Loading sentiment classifier onto device {self.device}...")
            model_tag = "j-hartmann/emotion-english-distilroberta-base"
            model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            # model = torch.compile(model)
            self.tokenizer = AutoTokenizer.from_pretrained(model_tag) # keep this for internal length checks later
            model_pipeline = pipeline("text-classification", model=model, tokenizer=self.tokenizer, device=self.device) # this'll spit out some warning, which we can ignore :D
            #model_pipeline = pipeline("text-classification", model=model_tag, return_all_scores=True, device=self.device)
            self.model = model_pipeline
            self.max_len = self.model.tokenizer.model_max_length - self.model.tokenizer.num_special_tokens_to_add()
            print("Loading completed!")
        self.tokenizer_cache = {} # for rechunking only
        return self.model

    @beartype
    def maybe_remove_chinese(self, text: str): # this happens in some models and really messes up the classifier -- just drop 
        if not self.CHINESE_PATTERN.search(text):  
            return text
        print("Warning: Chinese characters detected. Removing to avoid tokenization issues. This may result in inaccurate results.")
        return self.CHINESE_PATTERN.sub("", text)
    
    @beartype
    def __call__(self, text: str):        
        text = self.maybe_remove_chinese(text)
        sentences = sent_tokenize(text) # this model is trained at the sentence/utterance level, so break it up this way
        sentiment_by_sentences = []
        for sent in sentences: # TODO: pre-process sentences via length-check, then batch inference?
            sent = sent.strip()

            est_sentence_length = len(self.model.tokenizer(sent)["input_ids"]) # BOS, EOS
            if est_sentence_length > self.max_len: 
                print(f"Extra-long sentence detected (len: {est_sentence_length}), which exceeds model maximum length of {self.max_len}. Chunking sentence using the tokenizer. Note that this is expensive since the model tokenizer does not produce a one-to-one mapping of words to embeddings, and should be avoided when possible.")      
                
                chunks = self._rechunk(sent)
                for chunk in chunks:
                    n_tokens = len(self.model.tokenizer(chunk)["input_ids"])
                    if n_tokens > self.max_len:
                        warnings.warn(f"Chunk still exceeds maximum length ({n_tokens} > {self.max_len}). Please raise an issue; the `_rechunk` method likely needs to be redesigned. Full chunk:\n{chunk}")
                        chunk_results = [
                            {'label': 'surprise', 'score': -1.}, 
                            {'label': 'anger', 'score': -1.}, 
                            {'label': 'neutral', 'score': -1.}, 
                            {'label': 'disgust', 'score': -1.}, 
                            {'label': 'sadness', 'score': -1.}, 
                            {'label': 'fear', 'score': -1.}, 
                            {'label': 'joy', 'score': -1.}
                        ]
                        sentiment_by_sentences.append(pd.DataFrame(chunk_results))
                        continue
                    with torch.no_grad():
                        print("Tokens in chunk:", n_tokens)
                        chunk_results = self.model(chunk, top_k=None)
                    print(chunk_results)
                    sentiment_by_sentences.append(pd.DataFrame(chunk_results))
            else: # proceed normally
                with torch.no_grad():
                    sent_results = self.model(sent, top_k=None) 
                sentiment_by_sentences.append(pd.DataFrame(sent_results))

        if len(sentences) > len(sentiment_by_sentences): # if <, then chunking was used
            print(f"Not all sentences were converted successfully ({len(sentiment_by_sentences)}/{len(sentences)}). Consider double-checking this data point.")
        emotion_df = pd.concat(sentiment_by_sentences, keys=range(len(sentiment_by_sentences)))
        return emotion_df.groupby('label').mean()


class DetoxifyModel(Model):
    @beartype
    def load(self, force_reload: Optional[bool] = False):
        if self.model is None or force_reload:
            print("Loading toxicity classifier...")
            self.model = Detoxify('original', device=self.device)
            self.max_len = self.model.tokenizer.model_max_length - self.model.tokenizer.num_special_tokens_to_add()
        self.tokenizer_cache = {} # for rechunking only
        return self.model

    @beartype
    def __call__(self, text: str):
        sentences = sent_tokenize(text)
        toxicity_by_sent = []
        for sent in sentences:
            input_length = len(self.model.tokenizer(sent)["input_ids"])
            max_len = self.model.tokenizer.model_max_length
            if input_length <= max_len:
                toxicity = self.model.predict(sent)
                toxicity_by_sent.append(toxicity)
            else:
                print(f"Extra-long sentence detected (len: {input_length}), which exceeds model maximum length of {max_len}. Chunking sentence using the tokenizer. Note that this is expensive since the model tokenizer does not produce a one-to-one mapping of words to embeddings, and should be avoided when possible.")      
                    

                chunks = self._rechunk(sent)
                for chunk in chunks:
                    toxicity = self.model.predict(chunk)
                    toxicity_by_sent.append(toxicity)
        return pd.DataFrame(toxicity_by_sent).mean(axis=0)



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
        politeness_df = pd.DataFrame([politeness_model.transform_utterance(sent, spacy_nlp=self.get_spacy_model()).meta["politeness_strategies"] for sent in sentences])  # TODO: handle empty strings as edge cases
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
    
@GoalFactory.register_goal("positive_emotion")
class PositiveEmotion(Goal):
    # From Empath; Fast et al., (2016). We anchor to positive_emotion since it is also in LIWC (Pennebaker et al., '01).
    _empath_engine = None
    _empath_pos_categories = ["positive_emotion"] # could consider adding "optimism" or "achievement" later

    @classmethod
    def get_empath_engine(cls):
        if cls._empath_engine is None:
            cls._empath_engine = Empath()
        return cls._empath_engine

    @beartype
    def __call__(self, text: str):
        engine = self.get_empath_engine()
        result = engine.analyze(text, categories=self._empath_pos_categories, normalize=True)
        return np.array([result[cat] for cat in self._empath_pos_categories]).mean()

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
    #GoalFactory.get_default("politeness"),
    #GoalFactory.get_default("anger"),
    #GoalFactory.get_default("disgust"),
    #GoalFactory.get_default("fear"),
    #GoalFactory.get_default("joy"),
    #GoalFactory.get_default("sadness"),
    #GoalFactory.get_default("surprise"),
    GoalFactory.get_default("textual_diversity"),
    GoalFactory.get_default("text_length"),
    #GoalFactory.get_default("positive_emotion"),
    GoalFactory.get_default("formality"),
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

