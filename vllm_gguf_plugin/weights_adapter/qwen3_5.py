# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gguf
import torch
from huggingface_hub import hf_hub_download
from vllm.logger import init_logger
from vllm.model_executor.models.utils import WeightsMapper

from ..gguf_files import GGUFModelFiles
from ..gguf_utils import maybe_patch_hf_config_from_gguf
from ..quantization.config import GGUFConfig
from ..quantization.linear import GGUFLinearMethod
from ..quantization.vocal_embeds import GGUFEmbeddingMethod
from ..weight_utils import get_gguf_tensor_names, split_stacked_experts
from .base import (
    BaseGGUFWeightsAdapter,
    GGUFLoadPlan,
    GGUFWeight,
    ModelLoadSource,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig

logger = init_logger(__name__)

QWEN35_MODEL_TYPES = (
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
)
QWEN35_MOE_MODEL_TYPES = ("qwen3_5_moe", "qwen3_5_moe_text")
QWEN35_MTP_MODEL_TYPES = ("qwen3_5_mtp", "qwen3_5_moe_mtp")
QWEN35_MOE_MTP_MODEL_TYPES = ("qwen3_5_moe_mtp",)
QWEN35_ARCHITECTURES = {
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_text": "Qwen3_5ForConditionalGeneration",
    "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
    "qwen3_5_moe_text": "Qwen3_5MoeForConditionalGeneration",
}

QWEN35_ATTN_SUBSTR: dict[str, str] = {
    "attn_norm.": "input_layernorm.",
    "post_attention_norm.": "post_attention_layernorm.",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "attn_qkv.": "linear_attn.in_proj_qkv.",
    "attn_gate.": "linear_attn.in_proj_z.",
    "ssm_alpha.": "linear_attn.in_proj_a.",
    "ssm_beta.": "linear_attn.in_proj_b.",
    "ssm_conv1d.": "linear_attn.conv1d.",
    "ssm_norm.": "linear_attn.norm.",
    "ssm_out.": "linear_attn.out_proj.",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a.weight": "linear_attn.A_log",
    "ssm_a": "linear_attn.A_log",
}

QWEN35_MOE_SUBSTR: dict[str, str] = {
    "ffn_gate_inp_shexp.": "mlp.shared_expert_gate.",
    "ffn_gate_inp.": "mlp.gate.",
    "ffn_gate_exps.": "mlp.experts.0.gate_proj.",
    "ffn_up_exps.": "mlp.experts.0.up_proj.",
    "ffn_down_exps.": "mlp.experts.0.down_proj.",
    "ffn_gate_shexp.": "mlp.shared_expert.gate_proj.",
    "ffn_up_shexp.": "mlp.shared_expert.up_proj.",
    "ffn_down_shexp.": "mlp.shared_expert.down_proj.",
}

QWEN35_DENSE_SUBSTR: dict[str, str] = {
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
}

_MTP_NORM_SUFFIXES = (
    "norm.weight",
    "norm_embedding.weight",
    "norm_hidden.weight",
)


@dataclass(frozen=True, slots=True)
class _GDNLayout:
    num_key_heads: int
    value_head_repeat: int
    key_head_dim: int
    value_head_dim: int


class _Qwen35OutProjGGUFLinearMethod(GGUFLinearMethod):
    """Qwen GDN output projection over its packed GGUF column order."""

    def __init__(self, quant_config: GGUFConfig, repeat: int, head_dim: int):
        super().__init__(quant_config)
        self.repeat = repeat
        self.head_dim = head_dim

    def reorder_input(self, x: torch.Tensor) -> torch.Tensor:
        num_value_heads, remainder = divmod(x.shape[-1], self.head_dim)
        if remainder or num_value_heads % self.repeat:
            raise ValueError(
                "Cannot reorder Qwen3.5 GDN input shape "
                f"{tuple(x.shape)} with repeat={self.repeat}, "
                f"head_dim={self.head_dim}"
            )
        num_key_heads = num_value_heads // self.repeat
        shape = (*x.shape[:-1], num_key_heads, self.repeat, self.head_dim)
        return x.reshape(shape).transpose(-3, -2).reshape(x.shape).contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().apply(layer, self.reorder_input(x), bias)


def _enable_gguf_embedding(layer, quant_config: GGUFConfig) -> None:
    """Give Qwen's unquantized-by-construction embedding packed GGUF params."""
    if isinstance(layer.quant_method, GGUFEmbeddingMethod):
        return

    layer.register_parameter("weight", None)
    quant_method = GGUFEmbeddingMethod(quant_config)
    layer.quant_method = quant_method
    quant_method.create_weights(
        layer,
        layer.embedding_dim,
        [layer.num_embeddings_per_partition],
        layer.embedding_dim,
        layer.num_embeddings_padded,
        params_dtype=layer.params_dtype,
        weight_loader=layer.weight_loader,
    )


def _find_nextn_block_index(gguf_files: list[str]) -> int | None:
    """Return the first Qwen GGUF block containing an MTP/nextn layer."""
    for gguf_file in gguf_files:
        for tensor in gguf.GGUFReader(gguf_file).tensors:
            if match := re.match(r"blk\.(\d+)\.nextn\.", tensor.name):
                return int(match.group(1))
    return None


def _prefetch_mtp_weights(
    model_config: ModelConfig,
    download_dir: str | None,
) -> str | None:
    """Download only HF safetensors shards containing Qwen MTP parameters."""
    repo_id = str(model_config.model)
    if os.path.isdir(repo_id):
        return None
    try:
        index_path = hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            cache_dir=download_dir,
            revision=model_config.revision,
        )
    except Exception as error:
        logger.info("No MTP shard index for %s (%s)", repo_id, error)
        return None
    with open(index_path) as index_file:
        weight_map = json.load(index_file)["weight_map"]
    shards = sorted(
        {shard for name, shard in weight_map.items() if "mtp" in name.lower()}
    )
    if not shards:
        return None
    logger.info("Prefetching %d MTP shard(s): %s", len(shards), shards)
    for shard in shards:
        hf_hub_download(
            repo_id,
            shard,
            cache_dir=download_dir,
            revision=model_config.revision,
        )
    return os.path.dirname(index_path)


def qwen35_layer_substr(is_moe: bool) -> dict[str, str]:
    """Within-layer renames shared by the target and MTP draft."""
    mlp_substr = QWEN35_MOE_SUBSTR if is_moe else QWEN35_DENSE_SUBSTR
    return QWEN35_ATTN_SUBSTR | mlp_substr


def build_qwen35_text_mapper(is_multimodal: bool, is_moe: bool) -> WeightsMapper:
    backbone_prefix = "model.language_model." if is_multimodal else "model."
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": backbone_prefix + "embed_tokens.",
            "blk.": backbone_prefix + "layers.",
            "output_norm.": backbone_prefix + "norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=qwen35_layer_substr(is_moe),
    )


def build_qwen35_vision_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_prefix={
            "v.blk.": "model.visual.blocks.",
            "v.patch_embd.": "model.visual.patch_embed.proj.",
            "v.position_embd.": "model.visual.pos_embed.",
            "v.post_ln.": "model.visual.merger.norm.",
            "mm.0.": "model.visual.merger.linear_fc1.",
            "mm.2.": "model.visual.merger.linear_fc2.",
        },
        orig_to_new_substr={
            "attn_qkv.": "attn.qkv.",
            "attn_out.": "attn.proj.",
            "ffn_up.": "mlp.linear_fc1.",
            "ffn_down.": "mlp.linear_fc2.",
            "ln1.": "norm1.",
            "ln2.": "norm2.",
        },
    )


def build_qwen35_mtp_mapper(block_index: int, is_moe: bool) -> WeightsMapper:
    block = f"blk.{block_index}."
    return WeightsMapper(
        orig_to_new_prefix={
            f"{block}nextn.eh_proj.": "mtp.fc.",
            f"{block}nextn.enorm.": "mtp.pre_fc_norm_embedding.",
            f"{block}nextn.hnorm.": "mtp.pre_fc_norm_hidden.",
            f"{block}nextn.shared_head_norm.": "mtp.norm.",
            block: "mtp.layers.0.",
        },
        orig_to_new_substr=qwen35_layer_substr(is_moe),
    )


def _map_tensor_name(mapper: WeightsMapper, name: str) -> str | None:
    mapped = mapper.apply_list([name])[0]
    return mapped if mapped != name else None


def _gdn_layout(model_config: ModelConfig) -> _GDNLayout | None:
    text_config = model_config.hf_config.get_text_config()
    num_key_heads = getattr(text_config, "linear_num_key_heads", 0) or 0
    num_value_heads = getattr(text_config, "linear_num_value_heads", 0) or 0
    if not num_key_heads or not num_value_heads:
        return None
    repeat, remainder = divmod(num_value_heads, num_key_heads)
    if remainder or repeat <= 1:
        return None
    return _GDNLayout(
        num_key_heads=num_key_heads,
        value_head_repeat=repeat,
        key_head_dim=text_config.linear_key_head_dim,
        value_head_dim=text_config.linear_value_head_dim,
    )


class Qwen35GGUFAdapter(BaseGGUFWeightsAdapter):
    """Qwen3.5/3.6 dense, MoE, and multimodal GGUF adapter."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN35_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        return QWEN35_ARCHITECTURES.get(config.model_type)

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
        architecture = self.architecture(patched) or self.architecture(hf_config)
        if files.mm_proj is not None and getattr(patched, "vision_config", None) is None:
            if patched.model_type in ("qwen3_5", "qwen3_5_text"):
                from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

                patched = Qwen3_5Config(
                    text_config=patched.to_dict(),
                    architectures=[architecture],
                )
            else:
                from vllm.transformers_utils.configs.qwen3_5_moe import (
                    Qwen3_5MoeConfig,
                )

                patched = Qwen3_5MoeConfig(
                    text_config=patched.to_dict(),
                    architectures=[architecture],
                )
        if (
            getattr(patched, "vision_config", None) is not None
            and files.mm_proj is None
        ):
            raise RuntimeError(
                "Could not find mm_proj for multimodal Qwen3.5/3.6 GGUF. "
                "Place *mmproj*.gguf beside the backbone or pass "
                "model_loader_extra_config={'mm_proj': ...}."
            )
        if architecture is not None:
            patched.architectures = [architecture]
        return patched

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        config = model_config.hf_config
        is_multimodal = files.mm_proj is not None
        is_moe = config.model_type in QWEN35_MOE_MODEL_TYPES
        text_mapper = build_qwen35_text_mapper(is_multimodal, is_moe)
        vision_mapper = build_qwen35_vision_mapper()

        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.all_files)):
            if ".nextn." in name:
                continue
            mapper = vision_mapper if name.startswith(("v.", "mm.")) else text_mapper
            mapped = _map_tensor_name(mapper, name)
            if mapped is None:
                unmapped.append(name)
            else:
                name_map[name] = mapped
        if unmapped:
            logger.warning(
                "No HF name for %d Qwen3.5 GGUF tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def configure_model(
        self,
        model: torch.nn.Module,
        plan: GGUFLoadPlan,
        model_config: ModelConfig,
        quant_config: GGUFConfig,
    ) -> None:
        layout = _gdn_layout(model_config)
        if layout is not None:
            for name, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if not name.endswith("linear_attn.out_proj") or not isinstance(
                    quant_method, GGUFLinearMethod
                ):
                    continue
                replacement = _Qwen35OutProjGGUFLinearMethod(
                    quant_config,
                    layout.value_head_repeat,
                    layout.value_head_dim,
                )
                if hasattr(quant_method, "params_dtype"):
                    replacement.params_dtype = quant_method.params_dtype
                module.quant_method = replacement

        embed_name = plan.name_map.get("token_embd.weight")
        if embed_name is None:
            return
        embed_module = embed_name.removesuffix(".weight")
        if embed_module in plan.unquantized_modules:
            return

        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        embeddings = [
            module
            for name, module in model.named_modules()
            if name.endswith("embed_tokens")
            and type(module) is VocabParallelEmbedding
        ]
        if len(embeddings) != 1:
            raise RuntimeError(
                "Expected exactly one Qwen3.5 token embedding, found "
                f"{len(embeddings)}"
            )
        _enable_gguf_embedding(embeddings[0], quant_config)

    @staticmethod
    def _inverse_row_reorder(
        tensor: torch.Tensor,
        dim: int,
        num_key_heads: int,
        repeat: int,
        head_dim: int,
    ) -> torch.Tensor:
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        tensor = tensor.reshape(
            *shape[:dim],
            repeat,
            num_key_heads,
            head_dim,
            *shape[dim + 1 :],
        )
        tensor = tensor.transpose(dim, dim + 1)
        return tensor.reshape(*shape).contiguous()

    def _reorder_gdn(
        self,
        name: str,
        weight: torch.Tensor,
        layout: _GDNLayout,
    ) -> torch.Tensor | None:
        if name.endswith("qweight_type"):
            return None
        base = name.removesuffix(".qweight").removesuffix(".weight")
        inverse = self._inverse_row_reorder
        nk = layout.num_key_heads
        repeat = layout.value_head_repeat
        key_dim = layout.key_head_dim
        value_dim = layout.value_head_dim
        if base.endswith(".A_log"):
            return inverse(torch.log(-weight), 0, nk, repeat, 1)
        if base.endswith(".dt_bias"):
            return inverse(weight, 0, nk, repeat, 1)
        if base.endswith("linear_attn.in_proj_z"):
            return inverse(weight, 0, nk, repeat, value_dim)
        if base.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
            return inverse(weight, 0, nk, repeat, 1)
        if base.endswith("linear_attn.in_proj_qkv"):
            qk_rows = key_dim * nk * 2
            value = inverse(weight[qk_rows:], 0, nk, repeat, value_dim)
            return torch.cat([weight[:qk_rows], value], dim=0)
        if base.endswith("linear_attn.conv1d") and weight.dim() == 2:
            qk_rows = key_dim * nk * 2
            value = inverse(weight[qk_rows:], 0, nk, repeat, value_dim)
            return torch.cat([weight[:qk_rows], value], dim=0).unsqueeze(1)
        return None

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        layout = _gdn_layout(model_config)
        vision_config = getattr(model_config.hf_config, "vision_config", None)
        temporal_patch_size = getattr(vision_config, "temporal_patch_size", 1)

        def transformed() -> Iterable[GGUFWeight]:
            patch_embed_parts: dict[str, torch.Tensor] = {}
            for name, weight in weights:
                if layout is not None:
                    reordered = self._reorder_gdn(name, weight, layout)
                    if reordered is not None:
                        yield name, reordered
                        continue
                if name.endswith(".A_log"):
                    yield name, torch.log(-weight)
                    continue
                if (
                    name.endswith("norm.weight")
                    and not name.endswith("linear_attn.norm.weight")
                    and "visual" not in name
                ):
                    yield name, weight - 1
                    continue
                if temporal_patch_size > 1 and "patch_embed.proj.weight" in name:
                    base, split, _ = name.rpartition(".weight.")
                    key = f"{base}.weight" if split else name
                    patch_embed_parts[name] = weight
                    parts = [patch_embed_parts.get(key)] + [
                        patch_embed_parts.get(f"{key}.{index}")
                        for index in range(1, temporal_patch_size)
                    ]
                    if any(part is None for part in parts):
                        continue
                    yield key, torch.stack(parts, dim=2)
                    continue
                if "conv1d.weight" in name and weight.dim() == 2:
                    weight = weight.unsqueeze(1)
                elif (
                    name.endswith(".weight")
                    and weight.dim() == 1
                    and "norm" not in name
                ):
                    weight = weight.unsqueeze(0)
                yield name, weight

        yield from split_stacked_experts(transformed())


class Qwen35MtpGGUFAdapter(BaseGGUFWeightsAdapter):
    """Qwen3.5/3.6 single-block MTP draft stored in a GGUF nextn block."""

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN35_MTP_MODEL_TYPES

    def resolve_model_source(
        self,
        model_config: ModelConfig,
        target_model_config: ModelConfig,
        target_files: GGUFModelFiles,
        download_dir: str | None,
    ) -> ModelLoadSource:
        if _find_nextn_block_index(list(target_files.backbone)) is not None:
            model = target_model_config.model_weights or target_model_config.model
            return ModelLoadSource(model=str(model))

        logger.info(
            "Qwen3.5 GGUF has no nextn block; loading draft from HF repo %s",
            model_config.model,
        )
        snapshot = _prefetch_mtp_weights(model_config, download_dir)
        return ModelLoadSource(
            model=snapshot or str(model_config.model),
            load_format="auto",
            quantization=None,
        )

    def build_name_map(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
    ) -> dict[str, str]:
        block_index = _find_nextn_block_index(list(files.backbone))
        if block_index is None:
            raise RuntimeError(
                f"No MTP/nextn block in {files.backbone}; this GGUF cannot "
                "serve a speculative draft."
            )
        logger.info("Loading Qwen3.5 MTP draft from GGUF block %d", block_index)
        mapper = build_qwen35_mtp_mapper(
            block_index,
            is_moe=model_config.hf_config.model_type in QWEN35_MOE_MTP_MODEL_TYPES,
        )
        block_prefix = f"blk.{block_index}."
        name_map: dict[str, str] = {}
        unmapped: list[str] = []
        for name in sorted(get_gguf_tensor_names(files.backbone)):
            if not name.startswith(block_prefix):
                continue
            mapped = _map_tensor_name(mapper, name)
            if mapped is None:
                unmapped.append(name)
            else:
                name_map[name] = mapped
        if unmapped:
            logger.warning(
                "No HF name for %d Qwen3.5 MTP tensor(s), skipping: %s",
                len(unmapped),
                unmapped,
            )
        return name_map

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config: ModelConfig,
    ) -> Iterable[GGUFWeight]:
        del model_config

        def transformed() -> Iterable[GGUFWeight]:
            for name, weight in weights:
                if name.endswith(_MTP_NORM_SUFFIXES):
                    weight = weight - 1
                elif name.endswith(".weight") and weight.dim() == 1:
                    weight = weight.unsqueeze(0)
                yield name, weight

        yield from split_stacked_experts(transformed())
