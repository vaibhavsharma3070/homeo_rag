import requests
from typing import List
from abc import ABC, abstractmethod
from loguru import logger
from app.config import settings
import re


class LLMConnector(ABC):
    """Abstract base class for LLM connectors."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response from LLM (pass prompt as-is)."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is available."""
        pass


class OllamaConnector(LLMConnector):
    """Connector for Ollama local models with safe fallbacks."""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        # bumped defaults: 100s read timeout, 10s connect timeout
        self.max_timeout = getattr(settings, 'ollama_read_timeout', 100)
        self.connection_timeout = getattr(settings, 'ollama_connect_timeout', 10)

    def get_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.connection_timeout)
            if response.status_code == 200:
                models_data = response.json()
                return [m.get('name') for m in models_data.get('models', []) if m.get('name')]
            return []
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.connection_timeout)
            if response.status_code != 200:
                logger.warning(f"Ollama service not responding: {response.status_code}")
                return False

            models_data = response.json()
            available = [m.get('name') for m in models_data.get('models', []) if m.get('name')]
            if self.model not in available:
                logger.warning(f"Model '{self.model}' not found. Available models: {available}")
                return False

            logger.info(f"Ollama service available with model: {self.model}")
            return True
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
            return False

    def generate_response(self, prompt: str) -> str:
        """
        Generate response using Ollama.
        Returns '' on failure/timeout → pipeline fallback will handle.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.05,
                "num_predict": 600,  # allow longer answers
                "num_ctx": 4096,     # allow bigger context
                "stop": ["\n\nQUESTION:", "\n\nAnswer:", "\n\nQ:"]
            }
        }

        try:
            logger.info(f"Sending request to Ollama with {self.max_timeout}s timeout")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(self.connection_timeout, self.max_timeout)
            )

            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""

            result = response.json() or {}
            text = (result.get('response') or "").strip()
            if not text:
                logger.warning("Empty response from Ollama")
                return ""

            cleaned = self._clean_response(text)
            print('res :-',cleaned)
            logger.info(f"Ollama response received: {len(cleaned)} chars")
            return cleaned

        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama timeout after {self.max_timeout}s: {e}")
            return ""
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama connection error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Light cleanup; do not alter meaning."""
        cleaned = ' '.join(response.split()).strip()
        for start in ["Answer:", "A:", "Context:", "QUESTION:", "Q:"]:
            if cleaned.lower().startswith(start.lower()):
                cleaned = cleaned[len(start):].lstrip(' :').strip()
        return cleaned

    def simple_summarize(self, context_text: str, query: str, max_sentences: int = 2) -> str:
        """Very basic keyword-based summarizer as fallback."""
        sentences = re.split(r'(?<=[.!?]) +', context_text)
        query_words = set(query.lower().split())

        ranked = []
        for s in sentences:
            score = sum(1 for w in query_words if w in s.lower())
            if score > 0:
                ranked.append((score, s))

        ranked.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s for _, s in ranked[:max_sentences]]

        return " ".join(top_sentences).strip() or context_text[:300] + "..."


class LLMFactory:
    """Factory for creating LLM connectors with fallback logic."""

    @staticmethod
    def create_connector() -> LLMConnector:
        if getattr(settings, 'llm_provider', 'ollama') == "ollama":
            ollama = OllamaConnector()
            if ollama.is_available():
                logger.info("Using Ollama connector")
                return ollama
            else:
                logger.warning("Ollama not available; using fallback Ollama connector")
        return OllamaConnector()
