# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ..gguf_files import GGUFModelFiles
from ..weight_utils import (
    get_gguf_tensor_names,
    get_gguf_unquantized_params,
    gguf_quant_weights_iterator_multi,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

    from ..quantization.config import GGUFConfig
    from ..quantization.layout import GGUFLinearInputTransform


GGUFWeight = tuple[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class GGUFLoadPlan:
    """Everything the common loader needs after model-specific preparation."""

    files: GGUFModelFiles
    name_map: dict[str, str]
    unquantized_modules: tuple[str, ...]
    selected_tensors: frozenset[str] | None = None
    linear_input_transforms: dict[str, GGUFLinearInputTransform] = field(
        default_factory=dict
    )


class BaseGGUFWeightsAdapter(ABC):
    """Model-specific GGUF name mapping and tensor transformation hooks."""

    @classmethod
    @abstractmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        """Return whether this adapter supports *config*."""

    @classmethod
    def architecture(cls, config: PretrainedConfig) -> str | None:
        """Return an architecture override required before model loading."""
        del config
        return None

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

    def prepare(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> GGUFLoadPlan:
        """Build an immutable load plan shared by all adapters."""
        model_config.hf_config = self.patch_hf_config(files, model_config.hf_config)

        text_config = model_config.hf_config.get_text_config()
        backbone_names = get_gguf_tensor_names(files.backbone)
        text_config.update(
            {"tie_word_embeddings": "output.weight" not in backbone_names}
        )

        name_map = self.build_name_map(files, model_config)
        selected = self.select_tensor_names(files, model_config)
        selected_tensors = frozenset(selected) if selected is not None else None
        unquantized_modules = self.get_unquantized_modules(
            files,
            name_map,
            selected_tensors,
        )
        linear_input_transforms = self.get_linear_input_transforms(
            files,
            model_config,
            name_map,
        )

        return GGUFLoadPlan(
            files=files,
            name_map=name_map,
            unquantized_modules=unquantized_modules,
            selected_tensors=selected_tensors,
            linear_input_transforms=linear_input_transforms,
        )

    def get_linear_input_transforms(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
        name_map: dict[str, str],
    ) -> dict[str, GGUFLinearInputTransform]:
        """Describe input layouts required by GGUF linear weights."""
        del files, model_config, name_map
        return {}

    def get_unquantized_modules(
        self,
        files: GGUFModelFiles,
        name_map: dict[str, str],
        selected_tensors: frozenset[str] | None,
    ) -> tuple[str, ...]:
        """Return mapped modules whose GGUF weights are already unquantized."""
        unquantized_tensors = set(get_gguf_unquantized_params(list(files.all_files)))
        if selected_tensors is not None:
            unquantized_tensors.intersection_update(selected_tensors)

        modules = {
            self.transform_module_name(mapped_name.removesuffix(".weight"))
            for gguf_name in unquantized_tensors
            if (mapped_name := name_map.get(gguf_name)) is not None
            and mapped_name.endswith(".weight")
        }
        return tuple(sorted(modules))

    def select_tensor_names(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> Iterable[str] | None:
        """Optionally restrict loading to a subset of raw GGUF tensor names."""
        del files, model_config
        return None

    def transform_module_name(self, module_name: str) -> str:
        """Apply name-only transformations needed before model initialization."""
        return module_name

    def iter_weights(
        self,
        plan: GGUFLoadPlan,
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Read, map, and transform weights described by *plan*."""
        weights = gguf_quant_weights_iterator_multi(
            list(plan.files.all_files),
            plan.name_map,
            selected_tensors=plan.selected_tensors,
        )
        yield from self.transform_weights(weights, model_config)

    def configure_model(
        self,
        model: torch.nn.Module,
        plan: GGUFLoadPlan,
        model_config: ModelConfig,
        quant_config: GGUFConfig,
    ) -> None:
        """Apply model-specific adjustments after init and before loading."""
        del model, plan, model_config, quant_config

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        """Apply model-specific transformations to mapped weights."""
        del model_config
        yield from weights
