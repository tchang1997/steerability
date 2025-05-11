from argparse import ArgumentParser
from pathlib import Path
import pickle
import re
import warnings

from beartype import beartype
import numpy as np
import pandas as pd

from beartype.typing import Dict, List
from typing import Any, Optional

"""
    These instruction generators are very tailored to the default goal-space, but it's 
    unclear how to refactor this: each goal might require special 'treatment' for each
    type of instruction. One possibility is to move the magic strings for the direct 
    prompt into some database file, which can then be extended.
"""
NO_EXPLAIN = " Respond with only the rewritten text and do not explain your response."
DISAMBIG = " You MUST not change anything else about the other parts of the text, even if it makes the rewritten text sound unnatural or otherwise awkward."
MORE_ADJ_PHRASES = ["harder to read", "use more diverse language", "longer", "more formal"]
LESS_ADJ_PHRASES = ["easier to read", "use less diverse language", "shorter", "more informal"]
GOAL_INDEX = ["reading_difficulty", "textual_diversity", "text_length", "formality"]

DEEPSEEK_SOMETIMES_SECTION_BREAK = "\n---\n\n"

@beartype
def get_instruction_generator(prompt_strategy: str, database: Optional[Any] = None, prompter_kwargs: Optional[Dict[str, Any]] = None):
    if prompt_strategy == "direct":
        inst_generator = DirectTemplateInstruction()
    elif prompt_strategy == "underspecified":
        inst_generator = DirectUnderspecifiedInstruction()
    elif prompt_strategy == "direct+inst": # bootstrap CoT instructions + sample
        inst_generator = DirectGroundedInstruction(database, **prompter_kwargs)
    elif prompt_strategy == "instruct":
        inst_generator = InstructionOnly(database, **prompter_kwargs) 
    elif prompt_strategy == "cot": 
        inst_generator = DirectCoTInstruction()
    else:
        raise ValueError(f"{prompt_strategy} is not an implemented instruction generator.")
    return inst_generator

def load_instruction_database(path: str):
    if Path(path).is_file():
        with open(path, "rb") as f:
            database = pickle.load(f)
            return database
    return None

class InstructionGenerator(object):
    def __init__(
        self,
        database: Optional[pd.DataFrame] = None,
    ):
        self.database = database # for NN-based instruction generation

    def sample_prompt(
        self,
        deltas: pd.DataFrame,
        targets: Optional[pd.DataFrame] = None,
        **kwargs,
    ): 
        """
            Formally, we define a prompting strategy as a family of conditional distributions of the form

            p ~ P_strategy( . | source_goals, target_goals) (or equivalently, targets and deltas for goals)

            Both deltas and targets should have shape (N, n_goals).
        """
        if targets is not None:
            if deltas.shape != targets.shape:
                raise ValueError("Shapes between deltas and targets mismatch.")

    def clean_response(
        self,
        resp: str,
    ):
        lines = resp.split("\n")
        if len(lines) == 1: 
            return resp # generally, line breaks will appear between "sure, here's XYZ" and the text IF applicable
        if "here" in lines[0].lower() and "rewritten" in lines[0].lower(): # hack -- empirically, all instances of an initial "ok, here's X" we've seen follow this format 
            resp_clean = "\n".join(lines[1:])
        else:
            resp_clean = resp

        if resp_clean.startswith(DEEPSEEK_SOMETIMES_SECTION_BREAK):
            resp_clean = resp_clean[len(DEEPSEEK_SOMETIMES_SECTION_BREAK):] 

        return resp_clean

class DirectTemplateInstruction(InstructionGenerator):
    def __init__(self, database: Optional[pd.DataFrame] = None):
        super().__init__(database)
        self.pos_phrases = np.array(MORE_ADJ_PHRASES)
        self.neg_phrases = np.array(LESS_ADJ_PHRASES)
        self.base_text = "Please rewrite the following, but make it "

    def sample_prompt(
            self,
            deltas: pd.DataFrame,
            targets: Optional[pd.DataFrame] = None,
            no_explain: Optional[bool] = True,
            disambig: Optional[bool] = False
        ):
        instructions = []
        goal_names = deltas.columns.str.replace("delta_", "")
        deltas = np.ma.array(deltas.values, mask=(deltas.values == 0) | np.isnan(deltas.values)) # mask = invalid value; i.e., goal is inactive

        pos_phrases_in_use = []
        neg_phrases_in_use = []
        for goal in goal_names:
            try:
                goal_idx = GOAL_INDEX.index(goal)
                pos_phrases_in_use.append(self.pos_phrases[goal_idx])
                neg_phrases_in_use.append(self.neg_phrases[goal_idx])
            except ValueError:
                raise ValueError(f"Goal {goal} not supported. Valid goals: {GOAL_INDEX}")
        masked_goal_array = np.ma.array(np.where(deltas > 0, pos_phrases_in_use, neg_phrases_in_use), mask=deltas.mask)
        for i in range(len(deltas)): # for each example
            intents = []
            for intent, val in zip(masked_goal_array[i].compressed(), deltas[i].compressed()): # for all active goals and deltas for example i...
                # intent is an adjectival phrase

                modifier = None
                if np.abs(val) > 0.5:
                    modifier = "much "
                elif np.abs(val) < 0.2:
                    modifier = "slightly "
                # else in [0.2, 0.5]:
                #   no modifier

                if modifier is not None:
                    if not intent.startswith("more") and "more" in intent:
                        intents.append(intent.replace("more", modifier + "more")) # e.g., "use more diverse language" => "use slightly more diverse language"
                    elif not intent.startswith("less") and "less" in intent:
                        intents.append(intent.replace("less", modifier + "less"))
                    else:
                        intents.append(modifier + intent) # e.g., "much happier,"" "slightly happier"
                else:
                    intents.append(intent)
            if len(intents) > 1:
                np.random.shuffle(intents)
                english_list = ", ".join(intents[:-1]) + ", and " + intents[-1] + "."
                instruction = self.base_text + english_list 
            elif len(intents) == 0:
                warnings.warn(f"All goals have deltas that are too small: {deltas[i].compressed()} for goals {masked_goal_array[i].compressed()}") 
                instruction = self.base_text + "as close to the original as possible." # this simply makes the prompt coherent English
            else: # length = 1
                instruction = self.base_text + intents[0] + "."
            if disambig:
                instruction += DISAMBIG
            if no_explain:
                instruction += NO_EXPLAIN # prompt hack :/

            instructions.append(instruction)
        return instructions

class DirectUnderspecifiedInstruction(InstructionGenerator):
    def __init__(self, database: Optional[pd.DataFrame] = None):
        super().__init__(database)
        self.base_text = [
                "Please rewrite the following, but make it better.",
                "Please rewrite the following, but improve it.",
                "Please rewrite the following, but with some improvements.",
                "Please rewrite the following, but make it higher-quality.",
                "Please rewrite the following, but with more polish.",
                "Please rewrite the following, but refine it.",
                "Please rewrite the following, but make it less bad.",
                "Please rewrite the following, but make it actually good.",
                "Please rewrite the following, but make it worthwhile.",
                "Please rewrite the following, but make it not suck.",
        ]

    def sample_prompt(self, deltas: np.ndarray, targets: np.ndarray, no_explain: Optional[bool] = True, disambig: Optional[bool] = False):
        instructions = np.random.choice(self.base_text, size=len(deltas))
        if disambig:
            instructions = [inst + DISAMBIG for inst in instructions]
        if no_explain:
            instructions = [inst + NO_EXPLAIN for inst in instructions]
        return instructions
    
class DirectCoTInstruction(DirectTemplateInstruction):
    def __init__(self, database: Optional[pd.DataFrame] = None):
        super().__init__(database)

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None, no_explain: Optional[bool] = True, disambig: Optional[bool] = False):
        prompts = [
            f"{p} Before outputting the rewritten text, propose and discuss a few concrete edits you might apply to this specific text using the following format and replacing the placeholders in []:\n\n## Edits\n\n[your proposed edits]\n\n## Rewritten text\n\n[your rewritten text]" \
            for p in super().sample_prompt(deltas, targets, no_explain=no_explain, disambig=disambig)]
        return prompts

    def clean_response(self, resp: str):
        resp = super().clean_response(resp)
        lines = resp.split("\n")
        if not re.match(r"^\W*edits?$", lines[0].strip(), re.IGNORECASE):
            print("No header detected in CoT response. Header:", lines[0])   # normal for reasoning models that already do this             
            print("Attempting to find rewritten text header...") 
        final_lines = []
        append = False
        for line in lines:
            if append:
                final_lines.append(line)
            if re.match(r"^\W*rewritten text\W*$", line.strip(), re.IGNORECASE):
                append = True
        if len(final_lines) == 0:
            print("Malformed CoT response, consider re-prompting if the original looks wrong.")
            print("Original (header preview):", resp[:500] + "...")
            print("Original: (ending previvew:", "..." + resp[-500:])
            return resp
        return "\n".join(final_lines).strip()


class InstructionSamplingMixin:
    def add_instructions_to_prompt(self, prompts: List[str], deltas, max_instructions_per_prompt: int, no_explain: Optional[bool] = False, disambig: Optional[bool] = False):
        final_prompts = []
        line_prefix = "\n\t- "
        for (_, delta), prompt in zip(deltas.iterrows(), prompts):
            deltas_to_key = tuple([f"{col}_{delta[col]}" for col in delta.index])
            curr_options = self.database[deltas_to_key]
            K = min(max_instructions_per_prompt, len(curr_options))
            sampled_inst = np.random.choice(curr_options, size=K, replace=False)
            instruction_str = line_prefix + line_prefix.join(sampled_inst)
            prompt_str = prompt + " Some ways that you can do so might include:" + instruction_str + "\n\n" 
            if disambig:
                prompt += DISAMBIG
            if no_explain:
                prompt += NO_EXPLAIN
            final_prompts.append(prompt_str)

        return final_prompts
    
class NullInstruction(InstructionGenerator):
    def __init__(self, database: Optional[pd.DataFrame] = None):
        super().__init__(database) 

    def sample_prompt(self, deltas: np.ndarray, targets: np.ndarray, no_explain: Optional[bool] = True, disambig: Optional[bool] = False):
        instructions = ["Please rewrite the following."] * len(deltas)
        if disambig:
            instructions = [inst + DISAMBIG for inst in instructions]
        if no_explain:
            instructions = [inst + NO_EXPLAIN for inst in instructions]
        return instructions

class InstructionOnly(NullInstruction, InstructionSamplingMixin):
    def __init__(self, database: Optional[pd.DataFrame] = None, n_instructions: Optional[int] = 3, seed: Optional[int] = 42):
        super().__init__(database)
        self.n_instructions = n_instructions
        self.seed = seed
        

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None, no_explain: Optional[bool] = False, disambig: Optional[bool] = False):
        prompt = super().sample_prompt(deltas, targets, no_explain=no_explain, disambig=disambig) # underspecified prompt
        np.random.seed(self.seed)
        final_prompts = self.add_instructions_to_prompt(prompt, deltas, self.n_instructions)
        return final_prompts 
    
class DirectGroundedInstruction(DirectTemplateInstruction, InstructionSamplingMixin):
    def __init__(self, database: Optional[pd.DataFrame] = None, n_instructions: Optional[int] = 3, seed: Optional[int] = 42):
        super().__init__(database)
        self.n_instructions = n_instructions
        self.seed = seed

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None, no_explain: Optional[bool] = False, disambig: Optional[bool] = False):
        prompt = super().sample_prompt(deltas, targets, no_explain=no_explain, disambig=disambig) # direct prompt
        np.random.seed(self.seed)
        final_prompts = self.add_instructions_to_prompt(prompt, deltas, self.n_instructions)
        return final_prompts
    
if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--probe", type=str, help="Probe (CSV) for testing instruction generator.")
    psr.add_argument("--instruction-path", type=str, default="./data/cot/llama3.1_8b_insts_extracted_cot.pkl")
    psr.add_argument("--prompt-strategy", type=str, default="direct", help="Instruction generator to test.")
    psr.add_argument("--seed", type=int)
    psr.add_argument("--no-explain", action="store_true")
    psr.add_argument("--disambig", action="store_true")
    args = psr.parse_args()

    with open(args.instruction_path, "rb") as f:
        instructions = pickle.load(f)

    inst_gen = get_instruction_generator(args.prompt_strategy, database=instructions, prompter_kwargs=dict(n_instructions=3)) # TODO: figure out how to pass some dummy default kwargs for applicable generators
    probe = pd.read_csv(args.probe, index_col=0).sample(n=1, random_state=args.seed)
    deltas = probe.filter(like="delta_", axis=1)
    targets = probe.filter(like="targets_", axis=1)
    test_prompt = inst_gen.sample_prompt(deltas, targets, disambig=args.disambig, no_explain=args.no_explain)
    
    print("Steerability probe:", args.probe)
    print("Prompting strategy:", args.prompt_strategy)
    print("=" * 40)
    print("Source text info:")
    print(probe.iloc[0])
    print()
    print("Source text preview:")
    print(probe.iloc[0]["text"][:1000])
    print()
    print("Prompt (generated):")
    print(*test_prompt)
