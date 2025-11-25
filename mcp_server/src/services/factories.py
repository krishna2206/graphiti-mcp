"""Factory classes for creating LLM, Embedder, and Database clients."""

from config.schema import (
    DatabaseConfig,
    EmbedderConfig,
    LLMConfig,
)

# Try to import FalkorDriver if available
try:
    from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: F401

    HAS_FALKOR = True
except ImportError:
    HAS_FALKOR = False

# Import Gemini clients
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.embedder.gemini import GeminiEmbedder
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.gemini_client import GeminiClient


def _validate_api_key(provider_name: str, api_key: str | None, logger) -> str:
    """Validate API key is present.

    Args:
        provider_name: Name of the provider (e.g., 'OpenAI', 'Anthropic')
        api_key: The API key to validate
        logger: Logger instance for output

    Returns:
        The validated API key

    Raises:
        ValueError: If API key is None or empty
    """
    if not api_key:
        raise ValueError(
            f'{provider_name} API key is not configured. Please set the appropriate environment variable.'
        )

    logger.info(f'Creating {provider_name} client')

    return api_key


class LLMClientFactory:
    """Factory for creating LLM clients based on configuration."""

    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        """Create a Gemini LLM client."""
        import logging

        logger = logging.getLogger(__name__)

        if not config.providers.gemini:
            raise ValueError('Gemini provider configuration not found')

        api_key = config.providers.gemini.api_key
        _validate_api_key('Gemini', api_key, logger)

        llm_config = GraphitiLLMConfig(
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return GeminiClient(config=llm_config)


class EmbedderFactory:
    """Factory for creating Embedder clients based on configuration."""

    @staticmethod
    def create(config: EmbedderConfig) -> EmbedderClient:
        """Create a Gemini Embedder client."""
        import logging

        logger = logging.getLogger(__name__)

        if not config.providers.gemini:
            raise ValueError('Gemini provider configuration not found')

        api_key = config.providers.gemini.api_key
        _validate_api_key('Gemini Embedder', api_key, logger)

        from graphiti_core.embedder.gemini import GeminiEmbedderConfig

        gemini_config = GeminiEmbedderConfig(
            api_key=api_key,
            embedding_model=config.model or 'models/text-embedding-004',
            embedding_dim=config.dimensions or 768,
        )
        return GeminiEmbedder(config=gemini_config)


class RerankerFactory:
    """Factory for creating Reranker (cross-encoder) clients based on configuration."""

    @staticmethod
    def create(config: LLMConfig) -> GeminiRerankerClient:
        """Create a Gemini Reranker client."""
        import logging

        logger = logging.getLogger(__name__)

        if not config.providers.gemini:
            raise ValueError('Gemini provider configuration not found')

        api_key = config.providers.gemini.api_key
        _validate_api_key('Gemini Reranker', api_key, logger)

        llm_config = GraphitiLLMConfig(
            api_key=api_key,
            model='gemini-2.5-flash-lite',  # Optimized for reranking
            temperature=0.0,  # Deterministic for reranking
        )
        return GeminiRerankerClient(config=llm_config)


class DatabaseDriverFactory:
    """Factory for creating Database drivers based on configuration.

    Note: This returns configuration dictionaries that can be passed to Graphiti(),
    not driver instances directly, as the drivers require complex initialization.
    """

    @staticmethod
    def create_config(config: DatabaseConfig) -> dict:
        """Create database configuration dictionary for FalkorDB."""
        if not HAS_FALKOR:
            raise ValueError(
                'FalkorDB driver not available in current graphiti-core version'
            )

        # Use FalkorDB config if provided, otherwise use defaults
        if config.providers.falkordb:
            falkor_config = config.providers.falkordb
        else:
            # Create default FalkorDB configuration
            from config.schema import FalkorDBProviderConfig

            falkor_config = FalkorDBProviderConfig()

        # Check for environment variable overrides (for CI/CD compatibility)
        import os
        from urllib.parse import urlparse

        uri = os.environ.get('FALKORDB_URI', falkor_config.uri)
        password = os.environ.get('FALKORDB_PASSWORD', falkor_config.password)

        # Parse the URI to extract host and port
        parsed = urlparse(uri)
        host = parsed.hostname or 'localhost'
        port = parsed.port or 6379

        return {
            'driver': 'falkordb',
            'host': host,
            'port': port,
            'password': password,
            'database': falkor_config.database,
        }
