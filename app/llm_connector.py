import requests
from typing import List
from abc import ABC, abstractmethod
from loguru import logger
from app.config import settings
import re
import google.generativeai as genai


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


class GeminiConnector(LLMConnector):
    """Connector for Google Gemini API."""

    def __init__(self):
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model
        self.model = None
        self._configure()

    def _configure(self):
        """Configure Gemini API with the API key."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini configured with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if Gemini service is available."""
        try:
            if not self.api_key or self.api_key == "your_gemini_api_key_here":
                logger.warning("Gemini API key not configured")
                return False

            if not self.model:
                self._configure()

            if not self.model:
                return False

            # Test with a simple prompt
            test_response = self.model.generate_content(
                "Hello",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=10,
                    temperature=0.1,
                )
            )
            
            if test_response and test_response.text:
                logger.info("Gemini service is available")
                return True
            
            return False

        except Exception as e:
            logger.warning(f"Gemini service not available: {e}")
            return False

    def generate_response(self, prompt: str) -> str:
        """
        Generate response using Gemini.
        Returns '' on failure → pipeline fallback will handle.
        """
        try:
            if not self.model:
                logger.error("Gemini model not initialized")
                return ""

            logger.info("Sending request to Gemini")
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                max_output_tokens=2048,
            )

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            if not response or not response.text:
                logger.warning("Empty response from Gemini")
                return ""

            text = response.text.strip()
            cleaned = self._clean_response(text)
            logger.info(f"Gemini response received: {len(cleaned)} chars")
            return cleaned

        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            return ""

    def _clean_response(self, response: str) -> str:
        """Light cleanup; do not alter meaning."""
        cleaned = ' '.join(response.split()).strip()
        
        # Strip common role/prefix tokens
        for start in ["Answer:", "A:", "Context:", "QUESTION:", "Q:", "Assistant:", "assistant:"]:
            if cleaned.lower().startswith(start.lower()):
                cleaned = cleaned[len(start):].lstrip(' :').strip()
        
        # Remove model role markers like <|assistant|> or <|user|>
        cleaned = re.sub(r"^<\|(?:assistant|user|system)\|>\s*", "", cleaned, flags=re.IGNORECASE)
        
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
        provider = getattr(settings, 'gemini_model', 'gemini-2.5-flash-lite').lower()
        
        if provider == "gemini-2.5-flash-lite":
            gemini = GeminiConnector()
            if gemini.is_available():
                logger.info("Using Gemini connector")
                return gemini
        
        else:
            logger.warning(f"Unknown provider '{provider}'; defaulting to Ollama")
            return False