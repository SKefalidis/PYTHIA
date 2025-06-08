import google.genai as genai
from google.genai import errors
from google.genai import types
from typing_extensions import override
from src.elelem.provider import ApiProvider
from src.elelem.errors import RateLimitError


class GoogleAIProvider(ApiProvider):
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = genai.Client(api_key="YOUR_GOOGLE_API_KEY")

    @override
    def _generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    seed=451, # 0451
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.OFF,
                        ),
                    ]
                ))
            generated = response.text
            return generated
        except errors.APIError as e:
            if e.code == 429:
                raise RateLimitError("Google API rate limit exceeded") from e