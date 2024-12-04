from argparse import ArgumentParser

import pandas as pd

# This is a utility script to convert results files from an older (non-public) version of this code into a steerability probe with the same format as the new repo.

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--file", type=str, required=True)
    psr.add_argument("--output", type=str, default="steerbench_converted.csv")
    args = psr.parse_args()
    
    result_df = pd.read_csv(args.file, index_col=0)
    result_subset = result_df[[
        # info columns
        'index', 'text', 'source', 'weights',
        # raw source goals
        'readability', 'politeness', 'anger', 'disgust', 'fear', 'joy',         'sadness', 'surprise', 'diversity', 'toxicity', 'severe_toxicity', 
        'obscene', 'threat', 'insult', 'identity_attack', 'text_length',
        # source normalized in goal-space 
        'source_readability', 'source_politeness', 'source_anger',
        'source_disgust', 'source_fear', 'source_joy', 'source_sadness', 
        'source_surprise', 'source_diversity', 'source_toxicity',
        'source_severe_toxicity', 'source_obscene', 'source_threat', 
        'source_insult', 'source_identity_attack', 'source_text_length',
        # requested deltas normalized in goal-space
        'delta_readability', 'delta_politeness', 'delta_anger', 
        'delta_disgust', 'delta_fear', 'delta_joy', 'delta_sadness',
        'delta_surprise', 'delta_diversity', 'delta_text_length',
        # target normalized in goal-space 
        'target_readability', 'target_politeness', 'target_anger', 
        'target_disgust', 'target_fear', 'target_joy', 'target_sadness',
        'target_surprise', 'target_diversity', 'target_text_length',
    ]]

    # rename columns
    colmap = {
        "index": "original_index",
        "weights": "sampling_weights",
    }
    goalmap = {
        "readability": "reading_difficulty",
        "diversity": "textual_diversity",
    }
    for col in result_subset.columns:
        for old_goal_name, new_goal_name in goalmap.items():
            if col.endswith(old_goal_name):
                colmap[col] = col.replace(old_goal_name, new_goal_name)
    result_final = result_subset.rename(columns=colmap)
    result_final.to_csv(args.output)

