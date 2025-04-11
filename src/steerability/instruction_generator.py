from argparse import ArgumentParser
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
# " You must keep the tone of the text and other aspects of the text the same as the original unless explicitly instructed otherwise."
#MORE_ADJ_PHRASES = ["harder to read", "more polite", "angrier", "sound more disgusted", "more fearful-sounding", "happier", "sadder", "sound more surprised", "use more diverse language", "more verbose"]
#LESS_ADJ_PHRASES = ["easier to read", "more rude", "less angry", "sound less disgusted", "less fearful-sounding", "less happy", "less sad", "sound less surprised", "use less diverse language", "more concise"]
#GOAL_INDEX = ["reading_difficulty", "politeness", "anger", "disgust", "fear", "joy", "sadness", "surprise", "textual_diversity", "text_length"]

MORE_ADJ_PHRASES = ["harder to read", "more polite", "use more diverse language", "longer", "more positive", "more formal"]
LESS_ADJ_PHRASES = ["easier to read", "less polite", "use less diverse language", "shorter", "less positive", "more informal"]
GOAL_INDEX = ["reading_difficulty", "politeness", "textual_diversity", "text_length", "positive_emotion", "formality"]

DEEPSEEK_SOMETIMES_SECTION_BREAK = "\n---\n\n"

@beartype
def get_instruction_generator(prompt_strategy: str, database: Optional[Any] = None, prompter_kwargs: Optional[Dict[str, Any]] = None):
    if prompt_strategy == "direct":
        inst_generator = DirectTemplateInstruction()
    elif prompt_strategy == "underspecified":
        inst_generator = DirectUnderspecifiedInstruction()
    elif prompt_strategy == "direct+inst": # bootstrap CoT instructions + sample
        inst_generator = DirectGroundedInstruction(database, **prompter_kwargs)
    elif prompt_strategy == "one_to_ten":
        inst_generator = DirectGranularTemplateInstruction()
    elif prompt_strategy == "instruct":
        inst_generator = InstructionOnly(database, **prompter_kwargs) 
    elif prompt_strategy == "cot": 
        inst_generator = DirectCoTInstruction()
    elif prompt_strategy == "goal_enum":
       inst_generator = GoalEnumeratedHyperSpecificInstruction()
    else:
        raise ValueError(f"{prompt_strategy} is not an implemented instruction generator.")
    return inst_generator

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
    
      
    

METRIC_NOUN_PHRASES = [
    "reading level",
    "politeness",
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "diversity of the text",
    "verbosity of the text"
]
class DirectGranularTemplateInstruction(InstructionGenerator):
    def __init__(self, database: Optional[pd.DataFrame] = None):
        super().__init__(database)
        self.base_text = "Please rewrite the following. Assume that each aspect of the text lies on a 10 point scale, where 1 represents the lowest possible level of that aspect, while 10 represents the highest possible level. Adjust the given aspects as follows:\n"
        self.metric_names = np.array(METRIC_NOUN_PHRASES)

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None, no_explain: Optional[bool] = True, disambig: Optional[bool] = False):
        instructions = []
        goal_names = deltas.columns.str.replace("delta_", "")
        delta_arr = np.ma.array(deltas.values, mask=(deltas.values == 0) | np.isnan(deltas.values)) # mask = invalid value; i.e., goal is inactive

        masked_goal_array = np.ma.array(
                np.repeat(self.metric_names[None, :], len(delta_arr), axis=0),
                mask=delta_arr.mask
            )
        for i in range(len(deltas)):
            intents = []
            for intent, val in zip(masked_goal_array[i].compressed(), delta_arr[i].compressed()):
                intent_str = None
                if val < 0:
                    intent_str = f"\t- Decrease "
                elif val > 0:
                    intent_str = f"\t- Increase "
                if intent_str is not None:
                    intent_str += f"the level of {intent} by "
                    n_levels = int(np.round(np.abs(val) * 10, 0))
                    intent_str += f"{n_levels} levels."
                    intents.append(intent_str)
            np.random.shuffle(intents)
            intent_list = "\n".join(intents)
            instruction = self.base_text + intent_list + "\n\n"
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

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None):
        prompts = [f"{p} Briefly discuss your proposed edits before providing the rewritten text, using the following format and replacing the placeholders in []:\n\n## Edits\n\n[your proposed edits]\n\n## Rewritten text\n\n[your rewritten text]" for p in super().sample_prompt(deltas, targets, no_explain=False)]
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
    
class GoalEnumeratedHyperSpecificInstruction(DirectTemplateInstruction):
    def __init__(self, database: Optional[pd.DataFrame] = None):
        super().__init__(database)

    def sample_prompt(
            self,
            deltas: np.ndarray,
            targets: Optional[np.ndarray] = None,
            no_explain: Optional[bool] = False,
            disambig: Optional[bool] = False,
            perm_seed: Optional[int] = 42,
        ):
        starting_prompt = super().sample_prompt(deltas, targets, no_explain=no_explain, disambig=disambig) # do I do feedback as a switch since it's a static string, or do the step-by-step as a static string?
        final_prompts = []
        goal_names = deltas.columns.str.removeprefix("delta_")
        np.random.seed(perm_seed)

        target_likert = (10 * targets).round().astype(int)
        source_likert = (10 * (targets - deltas.fillna(0).values)).round().astype(int)
        for i, prompt in enumerate(starting_prompt):
            inactive_goal_mask = deltas.iloc[i].isna()
            active_goal_mask = ~inactive_goal_mask
            active_goal_names = goal_names[active_goal_mask][np.random.permutation(active_goal_mask.sum())]
            inactive_goal_names = goal_names[inactive_goal_mask][np.random.permutation(inactive_goal_mask.sum())]

            example_prompt = prompt
            example_prompt += (
                " You MUST not change anything else about the other parts of the text, "
                "even if it makes the rewritten text sound unnatural or otherwise awkward.\n\n"
            )
            example_prompt += (
                "Specifically, you are being graded on your ability to preserve these aspects. "
                "I've provided a scale from zero to ten for reference.\n"
            )
            for goal in inactive_goal_names:
                goal_corrected = goal.replace("_", " ").capitalize()
                goal_likert = target_likert.iloc[i][f"target_{goal}"]
                example_prompt += f"* {goal_corrected}: (currently: {goal_likert}/10) \n"

            example_prompt += "\nYou are also being graded on your ability to modify these aspects:\n"
            for goal in active_goal_names:
                goal_corrected = goal.replace("_", " ").capitalize()
                goal_likert = target_likert.iloc[i][f"target_{goal}"]
                starting_goal_likert = source_likert.iloc[i][f"target_{goal}"]

                example_prompt += f"* {goal_corrected}: (currently: {starting_goal_likert}/10; needs to be {goal_likert}/10) \n"

            example_prompt += (
                "\nThink step-by-step about how to keep each of these aspects the same, "
                "or change them exactly as desired. Historically, you have tended to impose a moderating "
                "effect on text without being asked, and changed things like the textual "
                "diversity/length without being asked. Assume that if it is not explicitly mentioned in "
                "the prompt, you MUST not modify that aspect of text."
            )
            final_prompts.append(f"## Instructions\n\n{example_prompt}\n\n## Text to rewrite")
        return final_prompts


class InstructionSamplingMixin:
    def add_instructions_to_prompt(self, prompts: List[str], max_instructions_per_prompt: int, no_explain: Optional[bool] = False, disambig: Optional[bool] = False):
        final_prompts = []
        line_prefix = "\n\t- "
        for i, prompt in enumerate(prompts):
            curr_options = self.database[str(i)][0]
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


class InstructionOnly(DirectUnderspecifiedInstruction, InstructionSamplingMixin):
    def __init__(self, database: Optional[pd.DataFrame] = None, n_instructions: Optional[int] = 3):
        super().__init__(database)
        self.n_instructions = n_instructions
        

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None, no_explain: Optional[bool] = False, disambig: Optional[bool] = False):
        prompt = super().sample_prompt(deltas, targets, no_explain=no_explain, disambig=disambig) # underspecified prompt
        final_prompts = self.add_instructions_to_prompt(prompt, self.n_instructions)
        return final_prompts 
    
class DirectGroundedInstruction(DirectTemplateInstruction, InstructionSamplingMixin):
    def __init__(self, database: Optional[pd.DataFrame] = None, n_instructions: Optional[int] = 3):
        super().__init__(database)
        self.n_instructions = n_instructions

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None, no_explain: Optional[bool] = False, disambig: Optional[bool] = False):
        prompt = super().sample_prompt(deltas, targets, no_explain=no_explain, disambig=disambig) # direct prompt
        final_prompts = self.add_instructions_to_prompt(prompt, self.n_instructions)
        return final_prompts
    
if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--probe", type=str, help="Probe (CSV) for testing instruction generator.")
    psr.add_argument("--prompt-strategy", type=str, default="direct", help="Instruction generator to test.")
    args = psr.parse_args()

    inst_gen = get_instruction_generator(args.prompt_strategy) # TODO: figure out how to pass some dummy default kwargs for applicable generators
    probe = pd.read_csv(args.probe, index_col=0, nrows=1)
    deltas = probe.filter(like="delta_", axis=1)
    targets = probe.filter(like="targets_", axis=1)
    test_prompt = inst_gen.sample_prompt(deltas, targets)
    
    print("Steerability probe:", args.probe)
    print("Prompting strategy:", args.prompt_strategy)
    print("=" * 40)
    print("Source text info:")
    print(probe.iloc[0])
    print()
    print("Prompt (generated):")
    print(*test_prompt)
