import warnings

from beartype import beartype
from typing import Any, List, Optional

def clean_deepseek_thinks(s: str, think_tag: Optional[str] = "think") -> str:
    end_tag = f"</{think_tag}>"
    end_pos = s.find(end_tag)
    if end_pos == -1:
        return s
    start_of_remaining_text = end_pos + len(end_tag) + 1

    # Skip additional newlines
    while s[start_of_remaining_text] == '\n':
        start_of_remaining_text += 1

    return s[start_of_remaining_text:]

_cleaner_fns = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": clean_deepseek_thinks,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": clean_deepseek_thinks,
    "Qwen/Qwen3-4B": clean_deepseek_thinks, # they use the same thinking template
    "Qwen/Qwen3-32B": clean_deepseek_thinks,
    "Qwen/Qwen3-30B-A3B": clean_deepseek_thinks,
}

def is_chat_completion_format(completions: List[Any]):
    if all(isinstance(item, str) for item in completions):
        return False
    elif all(isinstance(item, list) for item in completions):
        if all(isinstance(completion[0], dict) and {"role", "content"}.issubset(completion[0].keys()) for completion in completions):
            return True
        else:
            return False
    else:
        warnings.warn("Completions were not strings or dictionaries in OpenAI chat-completion JSON format.")
        return False

@beartype
def clean_model_output(model_name: str, s: str, **kwargs) -> str:
    cleaner = _cleaner_fns.get(model_name, None)
    if cleaner is not None:
        return cleaner(s, **kwargs)
    else:
        return s
