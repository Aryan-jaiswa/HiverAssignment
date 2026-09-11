import os
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> dict:
        """
        Generates a JSON response from the LLM based on the prompt.
        """
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            
    def generate(self, prompt: str) -> dict:
        if not self.client:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please create a .env file.")
            
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            
            # Extract JSON text
            text = response.text
            # Sometimes LLMs wrap json in markdown blocks, clean it up
            if text.startswith("```json"):
                text = text.strip("```json").strip("```").strip()
            
            return json.loads(text)
        except Exception as e:
            # Safely handle API failures or JSON parsing errors
            print(f"[LLM Error]: {e}")
            return {}

class MockLLMProvider(LLMProvider):
    """Used for testing without hitting the API."""
    def generate(self, prompt: str) -> dict:
        if "escalate" in prompt.lower() and "mock_force" in prompt.lower():
            return {
                "reply": "",
                "confidence": 0.0,
                "needs_escalation": True,
                "reason": "Mock forced escalation."
            }
            
        return {
            "reply": "This is a mocked auto-handle reply.",
            "confidence": 0.9,
            "needs_escalation": False,
            "reason": ""
        }
