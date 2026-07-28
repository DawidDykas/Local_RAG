from urllib import response

import requests
from core.logger_config import logger

class OllamaClient:
    """
    Low-level client for Ollama API.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "gemma3:4b",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name


    def chat(
        self,
        messages: list,
        options: dict = None,
        json_mode: bool = False
    ) -> str:

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": options 
        }


        if json_mode:
            payload["format"] = "json"

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )

        logger.debug(f"Ollama API request payload: {payload}")



        
        response.raise_for_status()
        logger.debug(response.json())
        return response.json()["message"]["content"]


OllamaClientInstance = OllamaClient(base_url="http://ollama:11434", model_name="gemma3:4b")
