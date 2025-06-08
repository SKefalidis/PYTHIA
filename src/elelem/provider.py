import argparse
from abc import ABC, abstractmethod
from time import sleep
from typing_extensions import override
from src.engine.config import CONFIG
from src.elelem.errors import RateLimitError


class Provider(ABC):
    """Abstract base class for all providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0, return_usage = False) -> str:
        pass


class ApiProvider(Provider):
    """Base class for API-based providers."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.api_key = api_key
    
    @abstractmethod
    def _generate(self, prompt: str, max_tokens: int, temperature: float, return_usage: bool) -> str:
        pass
    
    @override
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0, return_usage = False) -> str:
        while True:
            try:
                generated = self._generate(prompt, max_tokens, temperature, return_usage)
                return generated
            except RateLimitError as e:
                print(f"Rate limit exceeded for {self.model_name}. Retrying...")
                print(f"Error: {e}")
                sleep(10)


class ProviderFactory:
    """Factory for creating Provider instances."""
    
    @staticmethod
    def list_providers() -> list[str]:
        return ["openai", "google", "litellm"]
    
    @staticmethod
    def create_provider(provider_name: str, model_name: str, api_key: str) -> Provider:
        if provider_name == "openai":
            from src.elelem.providers.openai import OpenAIProvider
            return OpenAIProvider(model_name, api_key)
        elif provider_name == "google":
            from src.elelem.providers.google import GoogleAIProvider
            return GoogleAIProvider(model_name, api_key)
        elif provider_name == "litellm":
            from src.elelem.providers.litellm import LiteLLMProvider
            return LiteLLMProvider(model_name, api_key)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
        
    @staticmethod
    def create_from_args(args) -> Provider:
        if not hasattr(args, "llm_provider") or not hasattr(args, "llm"):
            raise ValueError("Arguments must include llm_provider and llm")
        return ProviderFactory.create_provider(args.llm_provider, args.llm, args.api_key if hasattr(args, "api_key") else "")
    
    @staticmethod
    def create_from_config(config: CONFIG) -> Provider:
        return ProviderFactory.create_provider(config.get("llm_provider"), config.get("llm_model"), config.get("llm_api_key", ""))

    @staticmethod
    def fill_parse_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        providers = ProviderFactory.list_providers()
        llm_group = parser.add_argument_group("LLM Settings")
        llm_group.add_argument("--llm_provider", type=str, required=False, choices=providers, default="openai",
                              help="The AI provider to use. Default is 'openai'.")
        llm_group.add_argument("--llm_model", type=str, required=False, default="gpt-4.1-mini",
                              help="The name of the model to use. Default is 'gpt-4.1-mini'.")
        llm_group.add_argument("--llm_api_key", type=str, required=False,
                              help="The API key for the provider (if needed)")
        return llm_group
        