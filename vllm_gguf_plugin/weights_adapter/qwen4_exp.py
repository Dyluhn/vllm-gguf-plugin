# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..weight_utils import get_gguf_tensor_names
from .base import GGUFWeight
from .qwen3_5 import Qwen35GGUFAdapter, qwen35_layer_substr

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN4_EXP_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")
QWEN4_EXP_ARCHITECTURE = "Qwen4ExpForCausalLM"

QWEN4_EXP_SUBSTR: dict[str, str] = {
    "hc_attn_norm.": "attn_hyper_connection.hc_norm.",
    "hc_attn_down.": "attn_hyper_connection.input_mix_weight_down.",
    "hc_attn_up.": "attn_hyper_connection.input_mix_weight_up.",
    "hc_attn_inject.": "attn_hyper_connection.block_inject_weight.",
    "hc_ffn_norm.": "mlp_hyper_connection.hc_norm.",
    "hc_ffn_down.": "mlp_hyper_connection.input_mix_weight_down.",
    "hc_ffn_up.": "mlp_hyper_connection.input_mix_weight_up.",
    "hc_ffn_inject.": "mlp_hyper_connection.block_inject_weight.",
    "indexer.q_proj.": "self_attn.indexer.index_q_proj.",
    "indexer.k_proj.": "self_attn.indexer.index_k_proj.",
    "indexer.q_norm.": "self_attn.indexer.q_layernorm.",
    "indexer.k_norm.": "self_attn.indexer.k_layernorm.",
    "ple_key.": "ple.key_proj.",
    "ple_value.": "ple.value_proj.",
    "ple_norm_key.": "ple.norm_key.",
    "ple_norm_query.": "ple.norm_query.",
    "ple_norm_conv.": "ple.norm_conv.",
    "ple_conv1d.": "ple.conv1d.",
}

_INDEXER_PROJ_RE = re.compile(
    r"^(?P<prefix>.+\.self_attn\.indexer)\.index_(?P<part>[qk])_proj\."
    r"(?P<suffix>weight|qweight|qweight_type)$"
)


def build_qwen4_exp_text_mapper() -> WeightsMapper:
    """Map llama.cpp Qwen4Exp GGUF names to upstream checkpoint names."""
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": "model.embed_tokens.",
            "blk.": "model.layers.",
            "output_hc_norm.": "model.hyper_connection_mixer.hc_norm.",
            "output_hc_down.": ("model.hyper_connection_mixer.input_mix_weight_down."),
            "output_hc_up.": "model.hyper_connection_mixer.input_mix_weight_up.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=qwen35_layer_substr(is_moe=True) | QWEN4_EXP_SUBSTR,
    )


def _mapped_name(mapper: WeightsMapper, name: str) -> str | None:
    mapped = mapper.apply_list([name])[0]
    return mapped if mapped != name else None


def build_qwen4_exp_name_map(
    tensor_names: Iterable[str],
    text_config: PretrainedConfig,
) -> tuple[dict[str, str], list[str]]:
    """Build the Qwen4Exp map without instantiating the 176B HF model."""
    ple_layer_ids = tuple(int(layer_id) for layer_id in text_config.ple_layer_ids)
    if len(ple_layer_ids) != 1:
        raise ValueError(
            "Qwen4Exp GGUF loading currently requires exactly one PLE layer, "
            f"got {ple_layer_ids}"
        )
    ple_layer_index = ple_layer_ids[0] - 1
    if ple_layer_index < 0:
        raise ValueError(f"PLE layer ids are 1-based, got {ple_layer_ids[0]}")

    mapper = build_qwen4_exp_text_mapper()
    name_map: dict[str, str] = {}
    unmapped: list[str] = []
    for name in sorted(tensor_names):
        if name == "per_layer_token_embd.weight":
            name_map[name] = (
                f"model.layers.{ple_layer_index}.ple.ple_embedding."
                "ngram_embedding.weight"
            )
            continue
        mapped = _mapped_name(mapper, name)
        if mapped is None:
            unmapped.append(name)
        else:
            name_map[name] = mapped
    return name_map, unmapped


def merge_qwen4_exp_indexer_projections(
    weights: Iterable[GGUFWeight],
) -> Iterable[GGUFWeight]:
    """Reassemble the GGUF indexer Q/K tensors into vLLM's fused projection."""
    pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for name, weight in weights:
        match = _INDEXER_PROJ_RE.match(name)
        if match is None:
            yield name, weight
            continue
        key = (match.group("prefix"), match.group("suffix"))
        part = match.group("part")
        if part in pending.setdefault(key, {}):
            raise ValueError(f"Duplicate Qwen4Exp indexer {part} tensor: {name}")
        pending[key][part] = weight

    for (prefix, suffix), parts in sorted(pending.items()):
        if set(parts) != {"q", "k"}:
            raise ValueError(
                "Qwen4Exp GGUF indexer projection is incomplete for "
                f"{prefix}.{suffix}: found {sorted(parts)}"
            )
        q_weight = parts["q"]
        k_weight = parts["k"]
        output_name = f"{prefix}.index_qk_proj.{suffix}"
        if suffix == "qweight_type":
            if not torch.equal(q_weight, k_weight):
                raise ValueError(
                    "Qwen4Exp GGUF indexer Q/K projections use different "
                    f"quantization types at {prefix}"
                )
            yield output_name, q_weight
        else:
            if q_weight.ndim != k_weight.ndim:
                raise ValueError(
                    "Qwen4Exp GGUF indexer Q/K ranks differ at "
                    f"{prefix}: {q_weight.ndim} != {k_weight.ndim}"
                )
            yield output_name, torch.cat((q_weight, k_weight), dim=0)


class Qwen4ExpGGUFAdapter(Qwen35GGUFAdapter):
    """Text-only Qwen3.8-Flash-Next GGUF adapter."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN4_EXP_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        del config
        return QWEN4_EXP_ARCHITECTURE

    def patch_hf_config(
        self,
        files: GGUFModelFiles,
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        patched = maybe_patch_hf_config_from_gguf(
            files.primary_backbone,
            hf_config,
            mmproj_path=files.mm_proj,
        )
        patched.architectures = [QWEN4_EXP_ARCHITECTURE]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        tensor_names = get_gguf_tensor_names(files.backbone)
        name_map, unmapped = build_qwen4_exp_name_map(
            tensor_names,
            model_config.hf_config.get_text_config(),
        )
        if unmapped:
            logger.warning(
                "No HF name for %d Qwen4Exp GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def get_ple_offload_prefixes(
        self,
        model_config: ModelConfig,
    ) -> tuple[str, ...]:
        """Return the mapped Qwen4Exp n-gram table subtree."""
        text_config = model_config.hf_config.get_text_config()
        ple_layer_ids = tuple(int(layer_id) for layer_id in text_config.ple_layer_ids)
        if len(ple_layer_ids) != 1:
            raise ValueError(
                "Qwen4Exp GGUF loading currently requires exactly one PLE layer, "
                f"got {ple_layer_ids}"
            )
        return (f"model.layers.{ple_layer_ids[0] - 1}.ple.ple_embedding.",)

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        merged = merge_qwen4_exp_indexer_projections(weights)
        yield from super().transform_weights(merged, model_config)
