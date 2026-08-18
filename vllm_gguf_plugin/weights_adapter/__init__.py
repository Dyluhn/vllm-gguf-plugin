# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from ..gguf_files import GGUFModelFiles
from .base import BaseGGUFWeightsAdapter
from .diffusion import (
    DiffusionGGUFAdapter,
    Flux2KleinDiffusionGGUFAdapter,
    QwenImageDiffusionGGUFAdapter,
    ZImageDiffusionGGUFAdapter,
    get_diffusion_gguf_adapter,
)
from .gemma3 import Gemma3GGUFAdapter
from .olmoe import OLMoEGGUFAdapter
from .transformers import TransformersGGUFWeightsAdapter

_ADAPTER_REGISTRY: list[type[BaseGGUFWeightsAdapter]] = [
    Gemma3GGUFAdapter,
    OLMoEGGUFAdapter,
]


def get_weights_adapter(config) -> BaseGGUFWeightsAdapter:
    """Return the adapter for *config*, falling back to Transformers mappings."""
    for cls in _ADAPTER_REGISTRY:
        if cls.matches(config):
            return cls()
    return TransformersGGUFWeightsAdapter()


__all__ = [
    "BaseGGUFWeightsAdapter",
    "DiffusionGGUFAdapter",
    "Flux2KleinDiffusionGGUFAdapter",
    "GGUFModelFiles",
    "Gemma3GGUFAdapter",
    "OLMoEGGUFAdapter",
    "QwenImageDiffusionGGUFAdapter",
    "TransformersGGUFWeightsAdapter",
    "ZImageDiffusionGGUFAdapter",
    "get_diffusion_gguf_adapter",
    "get_weights_adapter",
]
