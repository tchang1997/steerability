import warnings

import numpy as np
import pandas as pd


from typing import Any, Callable, Dict, List, Optional

"""
    These instruction generators are very tailored to the default goal-space, but it's 
    unclear how to refactor this: each goal might require special 'treatment' for each
    type of instruction. One possibility is to move the magic strings for the direct 
    prompt into some database file, which can then be extended.
"""
NO_EXPLAIN = " Respond with only the rewritten text and do not explain your response."
DISAMBIG = " You must keep the tone of the text and other aspects of the text the same as the original unless explicitly instructed otherwise."
MORE_ADJ_PHRASES = ["harder to read", "more polite", "angrier", "sound more disgusted", "more fearful-sounding", "happier", "sadder", "sound more surprised", "use more diverse language", "more verbose"]
LESS_ADJ_PHRASES = ["easier to read", "more rude", "less angry", "sound less disgusted", "less fearful-sounding", "less happy", "less sad", "sound less surprised", "use less diverse language", "more concise"]


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
        inst_generator = InstructionOnly(database, **prompter_kwargs) # TODO: implement this. Should be a subclass of DirectUnderspecifiedInstruction
    elif prompt_strategy == "cot": 
        inst_generator = DirectCoTInstruction()
    else:
        raise ValueError(f"{mode} is not an implemented instruction generator.")
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
        resp: list[str],
    ):
        lines = resp[0].split("\n")
        if "here" in lines[0].lower() and "rewritten" in lines[0].lower():  
            resp_clean = "\n".join(lines[1:])
        else:
            resp_clean = resp[0]
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
        deltas = np.ma.array(deltas.values, mask=deltas.values == 0)
        masked_goal_array = np.ma.array(np.where(deltas > 0, self.pos_phrases, self.neg_phrases), mask=deltas.mask)
        for i in range(len(deltas)):
            intents = []
            for intent, val in zip(masked_goal_array[i].compressed(), deltas[i].compressed()):
                modifier = None
                if np.abs(val) > 0.5:
                    modifier = "much "
                elif np.abs(val) < 0.2:
                    modifier = "slightly "
                # o/w: no modifier

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
                instruction = self.base_text + intents[0]
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
    "", # toxic -- unused
    "",
    "",
    "",
    "",
    "",
    "verbosity of the text"
]
class DirectGranularTemplateInstruction(InstructionGenerator):
    def __init__(self, database: Optional[pd.DataFrame] = None):
        super().__init__(database)
        self.base_text = "Please rewrite the following. Assume that each aspect of the text lies on a 10 point scale, where 1 represents the lowest possible level of that aspect, while 10 represents the highest possible level. Adjust the given aspects as follows:\n"
        self.metric_names = np.array(METRIC_NOUN_PHRASES)

    def sample_prompt(self, deltas: np.ndarray, targets: Optional[np.ndarray] = None, no_explain: Optional[bool] = True):
        instructions = []
        for i in range(len(deltas)):
            intents = []
            for intent, val in zip(self.metric_names[~deltas[i].mask], deltas[i].compressed()):
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

    def sample_prompt(self, deltas: np.ndarray, targets: np.ndarray, no_explain: Optional[bool] = True):
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

    def clean_response(self, resp: list[str]):
        resp = super().clean_response(resp)
        lines = resp.split("\n")
        if not re.match(r"^\W*edits?\W*$", lines[0].strip(), re.IGNORECASE):
            print("No header detected in CoT response. Original:", s)                
            print("Attempting to find rewritten text header...") 
        final_lines = []
        append = False
        for line in lines:
            if append:
                final_lines.append(line)
            if re.match(r"^\W*rewritten text\W*$", line.strip(), re.IGNORECASE):
                append = True
        if len(final_lines) == 0:
            print("Malformed CoT response, consider re-prompting. Original:",resp)
            return resp
        return "\n".join(final_lines).strip()

class InstructionSamplingMixin:
    def add_instructions_to_prompt(self, prompts: List[str], max_instructions_per_prompt: int, no_explain: Optional[bool] = False):
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
        prompt = super().sample_prompt(deltas, targets, no_explain=no_explain, disambig=disambig) # underspecified prompt
        final_prompts = self.add_instructions_to_prompt(prompt, self.n_instructions)
        return final_prompts
    
