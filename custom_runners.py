import logging

from sammo import PROMPT_LOGGER_NAME
from sammo.base import LLMResult, Costs
from sammo.runners import BaseRunner, OpenAIChat, RetriableError, NonRetriableError
from sammo.utils import serialize_json

from typing import Optional, Union

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


class DeepInfraMixIn:
    BASE_URL = r"https://api.deepinfra.com/v1/inference/"

    def _get_headers(self):
        return {"Authorization": ("Bearer " + self._api_config["api_key"]), "Content-Type": "application/json"}
    
    def _rest_url(self):
        return f"{self.BASE_URL}{self._model_id}"


class DeepInfraChat(DeepInfraMixIn, BaseRunner):

    @classmethod
    def _get_equivalence_class(cls, model_id: str) -> str:
        return model_id.split("/")[-1]

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        randomness: Optional[float] = 0,
        seed: int = 0,
        priority: int = 0,
        system_prompt: Optional[str] = None, # keep for now -- may use later
        history: Optional[list[dict]] = None, # keep for now -- may use later
        json_mode: Optional[bool] = None, 
    ) -> LLMResult:
        formatted_prompt = f"<s>[INST] {prompt} [/INST]" # Mistral/Mixtral specific, possibly 
        request = dict(
            input=formatted_prompt,
            max_new_tokens=self._max_context_window or max_tokens,
            temperature=randomness,
            stop=["</s>"]
        )
        fingerprint = serialize_json({"seed": seed, "generative_model_id": self._equivalence_class, **request})
        return await self._execute_request(request, fingerprint, priority)
    
    @staticmethod
    def _extract_costs(json_data: dict) -> dict:
        return None # unknown how to extract this from DeepInfra for now 

    async def _call_backend(self, request: dict) -> dict:
        async with self._get_session() as session:
            async with session.post(
                self._rest_url(),
                json=request,
                headers=self._get_headers(),
            ) as response:
                text = await response.json()
                if response.status in [429, 500, 503]:
                    raise RetriableError(f"Server error: {response.status} {text}")
                elif response.status == 200:
                    return text
                else:
                    raise NonRetriableError(f"Server error: {response.status} {text}")

    def _to_llm_result(self, request: dict, json_data: dict, fingerprint: Union[str, bytes]) -> LLMResult:
        return LLMResult(
            json_data["results"][0]["generated_text"],
            costs=Costs(json_data["num_input_tokens"], json_data["num_tokens"]),
        )

