from beartype import beartype
from typing import Optional

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
}

@beartype
def clean_model_output(model_name: str, s: str, **kwargs) -> str:
    cleaner = _cleaner_fns.get(model_name, None)
    if cleaner is not None:
        return cleaner(s, **kwargs)
    else:
        return s
