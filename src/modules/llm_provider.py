# modules/llm_provider.py

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

class OllamaLLM(LLM):
    model: str = OLLAMA_MODEL
    base_url: str = OLLAMA_BASE_URL

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "ollama"
