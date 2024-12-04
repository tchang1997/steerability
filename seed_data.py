from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
import random
import re

from beartype import beartype
from datasets import load_dataset
import nltk
import pandas as pd
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from typing import Any, Optional, Union

@beartype
def describe_dataset(
        name: str,
        version: Optional[str] = None,
        split: Optional[str] = None,
        feature_col: Optional[str] = None
    ) -> tuple[str, str]:
    description = f"Dataset: {name}"
    if version:
        description += f"/{version}"
    if split:
        description += f"/{split}" 
    if feature_col:
        full_description = description + f"\nSource text from: {feature_col}"
    else:
        full_description = description
    return description, full_description

@beartype
def load_huggingface_dataset(
        name: str,
        split: str,
        feature_col: str,
        version: Optional[str] = None,
    ) -> tuple[str, Any]:

    try:
        if version is not None:
            ds = load_dataset(name, version, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(name, split=split, trust_remote_code=True)
    except Exception:
        print("An error occurred while loading the dataset. Currently, we only support HuggingFace datasets.")
    name, desc = describe_dataset(name, version=version, split=split, feature_col=feature_col)
    print(desc)
    return name, ds

@beartype
def process_dataframe(
        df: Any,
        feature_col: str,
        paragraph_chunksize: Optional[int] = None, # 50 is probably a good setting to keep things well below 2048 tokens
        max_words_per_doc: Optional[int] = 2048,
        word_overflow_behavior: Optional['str'] = 'drop',
        newlines_are_sentences: Optional[bool] = False, # some people on Reddit do not believe in using punctuation. This causes errors.
        filters: dict[str, Union[list[Any], Any]] = None,
        sample: Optional[int] = None, # these two kws are only used if we want to take a subsample of the dataset (for tractability)
        select_first: Optional[int] = None,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
    """
        This function provides a standardized processing pipeline for HuggingFace datasets, where all datasets pass through three stages:

        1. Filtering - selecting values that correspond to certain criterion
        2. Splitting - convert single examples into multiple (usually shorter) derivative examples
        3. Postprocessing - editing example-by-example

        In the future, we could imagine creating a pipeline-like class where we can specify filters, splitters, and post-processors dynamically.
    """

    print("======== STAGE 1/3 - FILTER ========")
    if filters is not None:
        # filters is a dict of key-value pairs -- columns to values
        prev_length = len(df[feature_col])
        for col, colvalue in filters.items():
            df = df.filter(lambda x: x[col] == colvalue)
            new_length = len(df[feature_col])

            print(f"\t- applied filter '{col} where {colvalue}': N = {prev_length} -> {new_length}")
            prev_length = new_length

    orig_length = len(df[feature_col])
    if sample is None:
        text_list = df[feature_col]
        print(f"\t- random subsample: PASSTHROUGH (N={orig_length})")
    else:
        if random_state is not None:
            random.seed(random_state)
        text_list = random.sample(df[feature_col], k=sample)
        print(f"\t- random subsample: N = {orig_length} -> {sample}")
    print("filtering completed")

    print("======== STAGE 2/3 - SPLIT ========")
    if paragraph_chunksize is not None:
        print(f"\t- paragraph_chunksize={paragraph_chunksize}: Splitting datasets into {paragraph_chunksize}-sentence paragraphs")
        chunked_text = []
        for chapter in tqdm(text_list,leave=False):
            sentences = nltk.sent_tokenize(chapter)
            new_paragraphs = [" ".join(sentences[i: i + paragraph_chunksize]) for i in range(0, len(sentences), paragraph_chunksize)]
            chunked_text.append(new_paragraphs)
        flattened_text = list(chain.from_iterable(chunked_text))
        final_df = pd.DataFrame(flattened_text, columns=["text"])
    else:
        print(f"\t- paragraph_chunksize=None: PASSTHROUGH")
        final_df = pd.DataFrame(text_list, columns=["text"])
    print("split phase completed")

    print("======== STAGE 3/3 - POSTPROCESS ========")
    if newlines_are_sentences:
        print("\t- newlines_are_sentences=True: treating newlines `\\n` as sentences")
        final_df["text"] = final_df["text"].progress_apply(lambda x: re.sub("(?:[^.!?\s])\r?\n+", ". ", x))
    else:
        print("\t- newlines_are_sentences=False: PASSTHROUGH")

    if max_words_per_doc is not None:
        tqdm.pandas(desc=f"Checking text lengths (max. {max_words_per_doc})", leave=False)
        lengths = final_df["text"].progress_apply(lambda x: len(nltk.word_tokenize(x)))
        orig_length = len(final_df)
        if word_overflow_behavior == "drop":
            final_df = final_df[lengths <= max_words_per_doc]
        else:
            raise NotImplementedError()

        print(f"\t- max_words_per_doc={max_words_per_doc}: enforced word limits with behavior `{word_overflow_behavior}`; N = {orig_length} -> {len(final_df)}")
    else:
        print("\t- max_words_per_doc=None: PASSTHROUGH")

    if select_first is not None:
        print(f"\t- select_first={select_first}: slicing first {select_first} points")
        final_df = final_df.iloc[:select_first]
    else:
        print("\t- select_first=None: PASSTHROUGH")
    print("postprocessing completed")
    return final_df 

@beartype
def create_steerability_probe(configs: list[dict], reload: Optional[list[str]] = []) -> pd.DataFrame:
    data = []
    for dataset_config in configs:
        name = dataset_config["name"]
        name_alt = name.replace("/", "_")
        target_path = Path.cwd() / f"data/_intermediate_{name_alt}.csv"
        if target_path.is_file() and name not in reload:
            print(f"Loading pre-processed dataset subset `{name}` at {target_path}")
            final_dataset = pd.read_csv(target_path, index_col=0, low_memory=False)
            print("Found", len(final_dataset), "examples")
        else:
            try:
                name, df = load_huggingface_dataset( # TODO: support other datasets, or raw CSV files
                    dataset_config["name"],
                    dataset_config["split"],
                    dataset_config["feature_col"],
                    dataset_config.get("version", None),
                )
                final_dataset = process_dataframe(
                    df,
                    dataset_config["feature_col"],
                    **dataset_config.get("kwargs", {}),
                )

                final_dataset["source"] = name
                final_dataset.to_csv(target_path)
            except Exception:
                import traceback
                print(f"Exception raised while processing dataset `{name}`. Original exception trace follows:")
                print(traceback.format_exc())
                continue
        data.append(final_dataset.reset_index(drop=True))
    return pd.concat(data, axis=0)

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--config", type=str, default="./config/seed_data/default_seed_data.yml", help="Config file for collecting seed data for steerability probe.")
    psr.add_argument("--reload", type=str, nargs="+", help="List of dataset IDs (HuggingFace) to forcibly reload.", default=[])
    args = psr.parse_args()

    tqdm.pandas()
    yaml = YAML(typ='safe')
    with open(args.config, "r") as f:
        dataset_config = yaml.load(f)
    final_probe = create_steerability_probe(dataset_config["datasets"], reload=args.reload).reset_index(drop=True)
    final_probe.to_csv(Path.cwd() / "data"/ dataset_config["save"])
