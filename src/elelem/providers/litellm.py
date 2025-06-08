import litellm
from typing import List, Union, Tuple, Optional
from typing_extensions import override
from src.elelem.provider import ApiProvider
from src.elelem.errors import RateLimitError

class LiteLLMProvider(ApiProvider):
    
    def __init__(self, model_name: str, api_key: str = None):
        super().__init__(model_name, api_key)
        # LiteLLM typically looks for environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY),
        # but we can store the passed key to inject it into the completion call if needed.
        self.api_key = api_key

    @override
    def _generate(self, prompt: str|List, max_tokens: int = 500, temperature: float = 0.0, return_usage: bool = False) -> Union[str, Tuple[str, object]]:
        try:
            # 1. Standardize Input to Messages
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [
                    {"role": "system",
                    "content": "You are a helpful assistant that tries its best to follow the instructions given by the user to generate a satisfying result. Be concise, helpful and try your best to do what the user requests."},
                    {"role": "user",
                    "content": prompt}
                ]
            
            # 2. Call LiteLLM (Unified Interface)
            # litellm.completion handles the translation between OpenAI format 
            # and other providers (Anthropic, Google, Azure, etc.) automatically.
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=451,
                api_key=self.api_key, 
                # Optional: prevents errors if you send OpenAI-specific params to other providers
                drop_params=True 
            )

            # 3. Extract Content
            # LiteLLM returns a response object that mimics the OpenAI structure
            content = response.choices[0].message.content

            if return_usage == False:
                return content
            return content, response.usage

        except litellm.exceptions.RateLimitError as e:
            # Map LiteLLM's rate limit exception to your internal error
            raise RateLimitError(f"LiteLLM ({self.model_name}) API rate limit exceeded") from e
        except Exception as e:
            # You may want to log the specific LiteLLM error here
            raise e