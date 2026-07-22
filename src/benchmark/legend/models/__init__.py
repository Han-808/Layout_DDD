"""Legend model adapters for the retired bbox-layout workflow."""

from benchmark.legend.models.base_model import BaseLayoutModel
from benchmark.legend.models.factory import MODEL_ADAPTERS, create_model
from benchmark.legend.models.mock_model import MockModel
from benchmark.legend.models.openai_compatible_model import OpenAICompatibleModel

__all__ = ["BaseLayoutModel", "MODEL_ADAPTERS", "MockModel", "OpenAICompatibleModel", "create_model"]
