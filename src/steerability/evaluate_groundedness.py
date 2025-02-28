from argparse import ArgumentParser
from functools import lru_cache
import os

from nltk.translate.meteor_score import single_meteor_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import spacy
from tqdm.auto import tqdm


EN_CORE_WEB_SM = spacy.load("en_core_web_sm", disable=['parser', 'textcat'])
FORBIDDEN_POS = ["AUX", "SPACE", "PUNCT", "DET", "CCONJ", "ADP", "ADV", "ADJ", "SCONJ", "SPART", "VERB"]

@lru_cache(maxsize=4096)
def debone(sent):
    # I don't know why I'm calling "only keep nouns and other entities" the "debone" function.
    try:
        doc = EN_CORE_WEB_SM(sent)
    except ValueError as e:
        print("Got ValueError when processing this sentence:", sent)
        print(e)
        return []
    return [tok.text for tok in doc if (tok.pos_ not in FORBIDDEN_POS) or tok.text in doc.ents]

def single_source_meteorite_score(df, index):
    assert len(df.loc[df["original_index"] == index, "text"].unique()) == 1
    source_text = df.loc[df["original_index"] == index, "text"].iloc[0]
    generated_texts = df.loc[df["original_index"] == index, "llm_response"]
    everything_else = df.loc[df["original_index"] != index, "llm_response"] # small sample. This is insane
    meteor_generated = [single_meteor_score(debone(source_text), debone(gen), alpha=0.5) for gen in tqdm(generated_texts, desc="METEOR, rewrites", leave=False)]
    meteor_everything_else = [single_meteor_score(debone(source_text), debone(other), alpha=0.5) for other in tqdm(everything_else, desc="METEOR, everything else", leave=False)]
    y = np.concatenate([np.ones_like(meteor_generated), np.zeros_like(meteor_everything_else)], axis=0)
    X = np.concatenate([meteor_generated, meteor_everything_else], axis=0).reshape(-1, 1)
    model = LogisticRegression()
    model.fit(X, y)
    preds = model.predict_proba(X)
    groundedness = preds[:len(meteor_generated), 1] / preds[:len(meteor_generated), 0] * len(meteor_everything_else) / len(meteor_generated) # P(score | rewrite of text) / P(score | not rewrite of same text) 
    # bayes rule -- need to make this adjustment so that >1 = more likely than not to be grounded and <1 is the opposite
    return groundedness

def meteorite_score(df):
    final_scores = np.concatenate([single_source_meteorite_score(df, idx) for idx in tqdm(df["original_index"].unique())], axis=0)
    df["grounding"] = final_scores
    return df

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--results", type=str, help="File to compute grounding scores for.")
    args = psr.parse_args()

    result_path = args.results.replace(".csv", "_grd.csv")
    if os.path.isfile(result_path):
        print("Skipping -- result exists at", result_path)

    df = pd.read_csv(args.results, index_col=0)
    df_with_ground = meteorite_score(df)
    df_with_ground.to_csv(result_path)
