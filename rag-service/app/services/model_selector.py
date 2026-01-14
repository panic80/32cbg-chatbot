"""Model selector service for fast/smart model routing.

This service manages the selection of appropriate LLM models based on
operation type (fast vs smart) and configuration.
"""

import json
import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import Provider

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: Provider
    model: str


@dataclass
class OperationModelAssignment:
    """Model type assignment for each operation."""
    response_generation: str = "smart"  # 'fast' or 'smart'
    hyde_expansion: str = "fast"
    query_rewriting: str = "fast"
    follow_up_generation: str = "fast"


class ModelSelector:
    """Selects the appropriate model based on operation type.

    This class manages fast/smart model configuration and routes
    operations to the appropriate model based on settings.
    """

    def __init__(self):
        """Initialize the model selector with default configuration."""
        # Default to settings-based configuration
        # Fast/auxiliary model is always gpt-5-mini via OpenAI
        self._fast_provider = Provider.OPENAI
        self._fast_model = settings.auxiliary_model
        self._smart_provider = Provider(settings.smart_model_provider)
        self._smart_model = settings.smart_model_name

        # Default operation assignments
        self._operations = OperationModelAssignment()

        # Try to load config from file if available
        self._load_config_from_file()

        logger.info(
            f"ModelSelector initialized - Fast: {self._fast_provider.value}/{self._fast_model}, "
            f"Smart: {self._smart_provider.value}/{self._smart_model}"
        )

    def _load_config_from_file(self):
        """Load configuration from JSON file if it exists."""
        config_path = os.getenv("MODEL_CONFIG_PATH")
        if not config_path:
            # Check default locations
            for path in [
                "/etc/cbthis/model-config.json",
                os.path.join(os.getcwd(), "server", "data", "model-config.json"),
            ]:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Load fast model config
                if "fastModel" in config:
                    fast = config["fastModel"]
                    self._fast_provider = Provider(fast.get("provider", "openai"))
                    self._fast_model = fast.get("model", settings.auxiliary_model)

                # Load smart model config
                if "smartModel" in config:
                    smart = config["smartModel"]
                    self._smart_provider = Provider(smart.get("provider", "openai"))
                    self._smart_model = smart.get("model", settings.smart_model_name)

                # Load operation assignments
                if "operationModels" in config:
                    ops = config["operationModels"]
                    self._operations = OperationModelAssignment(
                        response_generation=ops.get("responseGeneration", "smart"),
                        hyde_expansion=ops.get("hydeExpansion", "fast"),
                        query_rewriting=ops.get("queryRewriting", "fast"),
                        follow_up_generation=ops.get("followUpGeneration", "fast"),
                    )

                logger.info(f"Loaded model configuration from {config_path}")

            except Exception as e:
                logger.warning(f"Failed to load model config from {config_path}: {e}")

    def get_model_for_operation(self, operation: str) -> Tuple[Provider, str]:
        """Get provider and model for a specific operation.

        Args:
            operation: One of 'response', 'hyde', 'query_rewrite', 'followup'

        Returns:
            Tuple of (Provider, model_name)
        """
        # Map operation to assignment
        operation_map = {
            "response": self._operations.response_generation,
            "response_generation": self._operations.response_generation,
            "hyde": self._operations.hyde_expansion,
            "hyde_expansion": self._operations.hyde_expansion,
            "query_rewrite": self._operations.query_rewriting,
            "query_rewriting": self._operations.query_rewriting,
            "retrieval": self._operations.query_rewriting,
            "followup": self._operations.follow_up_generation,
            "follow_up_generation": self._operations.follow_up_generation,
        }

        designation = operation_map.get(operation.lower(), "smart")

        if designation == "fast":
            return self._fast_provider, self._fast_model
        else:
            return self._smart_provider, self._smart_model

    def get_fast_model(self) -> Tuple[Provider, str]:
        """Get the configured fast model."""
        return self._fast_provider, self._fast_model

    def get_smart_model(self) -> Tuple[Provider, str]:
        """Get the configured smart model."""
        return self._smart_provider, self._smart_model

    def update_config(
        self,
        fast_provider: Optional[str] = None,
        fast_model: Optional[str] = None,
        smart_provider: Optional[str] = None,
        smart_model: Optional[str] = None,
        operations: Optional[Dict[str, str]] = None
    ):
        """Update model configuration at runtime.

        Args:
            fast_provider: Provider for fast model
            fast_model: Model name for fast model
            smart_provider: Provider for smart model
            smart_model: Model name for smart model
            operations: Dict mapping operation names to 'fast' or 'smart'
        """
        if fast_provider:
            self._fast_provider = Provider(fast_provider)
        if fast_model:
            self._fast_model = fast_model
        if smart_provider:
            self._smart_provider = Provider(smart_provider)
        if smart_model:
            self._smart_model = smart_model

        if operations:
            self._operations = OperationModelAssignment(
                response_generation=operations.get("responseGeneration", self._operations.response_generation),
                hyde_expansion=operations.get("hydeExpansion", self._operations.hyde_expansion),
                query_rewriting=operations.get("queryRewriting", self._operations.query_rewriting),
                follow_up_generation=operations.get("followUpGeneration", self._operations.follow_up_generation),
            )

        logger.info(
            f"ModelSelector updated - Fast: {self._fast_provider.value}/{self._fast_model}, "
            f"Smart: {self._smart_provider.value}/{self._smart_model}"
        )

    def reload_config(self):
        """Reload configuration from file."""
        self._load_config_from_file()

    def get_config_summary(self) -> Dict[str, Any]:
        """Get current configuration summary."""
        return {
            "fastModel": {
                "provider": self._fast_provider.value,
                "model": self._fast_model,
            },
            "smartModel": {
                "provider": self._smart_provider.value,
                "model": self._smart_model,
            },
            "operations": {
                "responseGeneration": self._operations.response_generation,
                "hydeExpansion": self._operations.hyde_expansion,
                "queryRewriting": self._operations.query_rewriting,
                "followUpGeneration": self._operations.follow_up_generation,
            }
        }


# Global singleton instance
_model_selector: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """Get the global model selector instance."""
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelSelector()
    return _model_selector


def reload_model_selector():
    """Reload the model selector configuration."""
    global _model_selector
    if _model_selector:
        _model_selector.reload_config()
    else:
        _model_selector = ModelSelector()
