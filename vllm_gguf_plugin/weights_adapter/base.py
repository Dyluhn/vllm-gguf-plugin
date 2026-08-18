# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from ..gguf_files import GGUFModelFiles

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


GGUFWeight = tuple[str, torch.Tensor]


class BaseGGUFWeightsAdapter(ABC):
    """Model-specific GGUF name mapping and tensor transformation hooks."""

    @classmethod
    @abstractmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        """Return whether this adapter supports *config*."""

    @abstractmethod
    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        """Map raw GGUF tensor names to names accepted by the model."""

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        """Patch HF config before model init."""
        del files
        return hf_config

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Apply model-specific transformations to mapped weights."""
        del model_config
        yield from weights
