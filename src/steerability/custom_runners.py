import logging

from sammo import PROMPT_LOGGER_NAME
from sammo.base import LLMResult
from sammo.runners import OpenAIChat
from sammo.utils import serialize_json

from typing import Optional

prompt_logger = logging.getLogger(PROMPT_LOGGER_NAME)

class VLLMMixIn:
    BASE_URL = r"http://localhost:{}/v1"
    SUFFIX = "/chat/completions"

    @classmethod
    def _get_equivalence_class(cls, model_id: str) -> str:
        return model_id.split("/")[-1]

    def _rest_url(self):
        return f"{self.BASE_URL.format(self.port)}{self.SUFFIX}"
    

class VLLMOpenAIChat(VLLMMixIn, OpenAIChat):
    def __init__(self, *args, port: Optional[int] = 5000, **kwargs):
        super().__init__(*args, **kwargs)
        self.port = port

    async def generate_text(
        self,
        prompt: str,
        max_tokens: int | None = None,
        randomness: float | None = 0,
        seed: int = 0,
        priority: int = 0,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
        num_generations: int = 1,
        frequency_penalty: float = 0.0,
        min_p: float = 0.0,
        thinking_hard_switch: Optional[bool] = None, # if unset, will not add key AT ALL to ensure compat with non-thinking models
        json_mode: bool = False,
    ) -> LLMResult:
        messages = []
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                history = [x for x in history if x["role"] != "system"]
        if history is not None:
            messages = messages + history

        # check for images in prompt
        revised_prompt = self._post_process_prompt(prompt)
        messages += [{"role": "user", "content": revised_prompt}]

        request = dict(
            messages=messages,
            max_tokens=self._max_context_window or max_tokens,
            temperature=randomness,
            n=num_generations,
            frequency_penalty=frequency_penalty,
            min_p=min_p,
        )
        if thinking_hard_switch is not None:
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking_hard_switch}}
        if json_mode is True:
            request["response_format"] = {"type": "json_object"}
        fingerprint = serialize_json({"seed": seed, "generative_model_id": self._equivalence_class, **request})

        return await self._execute_request(request | {"model": self._model_id}, fingerprint, priority)
