from email import message
import openai
from typing import List, Optional
from openai.types.shared.reasoning import Reasoning
from openai.types.shared.reasoning_effort import ReasoningEffort
from typing_extensions import override
from src.elelem.provider import ApiProvider
from src.elelem.errors import RateLimitError


class OpenAIProvider(ApiProvider):
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.openai = openai.OpenAI(api_key=api_key)

    @override
    def _generate(self, prompt: str|List, max_tokens: int = 500, temperature: float = 0.0, return_usage = False) -> str:
        try:
            if self.model_name.startswith("gpt-5"):
                max_tokens = max_tokens * 3  # GPT-5 models need more tokens to support reasoning.
                response = self.openai.responses.create(
                    model=self.model_name,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "low"},
                )
                text = getattr(response, "output_text", None)
                if text is None:
                    # Fallback: attempt to extract first text part
                    try:
                        text = response.output[0].content[0].text
                    except Exception:
                        text = str(response)
                if return_usage == False:
                    return text
                return text, getattr(response, "usage", None)
            else:
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [
                        {"role": "system",
                        "content": "You are a helpful assistant that tries its best to follow the instructions given by the user to generate a satisfying result. Be concise, helpful and try your best to do what the user requests."},
                        {"role": "user",
                        "content": prompt}
                    ]
                
                response = self.openai.chat.completions.create(
                    seed=451, # 0451
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages
                )
                if return_usage == False:
                    return response.choices[0].message.content
                return response.choices[0].message.content, response.usage
        except openai.RateLimitError as e:
            raise RateLimitError("OpenAI API rate limit exceeded") from e