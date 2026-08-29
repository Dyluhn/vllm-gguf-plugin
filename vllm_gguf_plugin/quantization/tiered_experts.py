# R9V modification: Qwen3.8 Flash Next GGUF/ROCm integration.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .params import allocate_uva_host_empty

logger = logging.getLogger(__name__)

_MANIFEST_ENV = "RADIANCE_TIERED_EXPERT_MANIFEST"
_CACHE_SLOTS_ENV = "QWEN38_TIERED_EXPERT_CACHE_SLOTS"
_CACHE_RANKS_ENV = "QWEN38_TIERED_EXPERT_CACHE_RANKS"
_CACHE_ASYNC_ENV = "QWEN38_TIERED_EXPERT_CACHE_ASYNC"
_CACHE_POLICY_ENV = "QWEN38_TIERED_EXPERT_CACHE_POLICY"
_MAX_CACHE_SLOTS = 16
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")


def _dynamic_cache_slots(rank: int) -> int:
    raw_slots = os.environ.get(_CACHE_SLOTS_ENV, "0")
    try:
        slots = int(raw_slots)
    except ValueError as error:
        raise ValueError(f"{_CACHE_SLOTS_ENV} must be an integer") from error
    if not 0 <= slots <= _MAX_CACHE_SLOTS:
        raise ValueError(
            f"{_CACHE_SLOTS_ENV} must be between 0 and {_MAX_CACHE_SLOTS}"
        )
    try:
        ranks = {
            int(value.strip())
            for value in os.environ.get(_CACHE_RANKS_ENV, "1").split(",")
            if value.strip()
        }
    except ValueError as error:
        raise ValueError(
            f"{_CACHE_RANKS_ENV} must be a comma-separated list of TP ranks"
        ) from error
    return slots if rank in ranks else 0


def _async_cache_enabled() -> bool:
    value = os.environ.get(_CACHE_ASYNC_ENV, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{_CACHE_ASYNC_ENV} must be 0 or 1")
    return value == "1"


def _cache_policy() -> str:
    value = os.environ.get(_CACHE_POLICY_ENV, "second_touch_rr")
    if value not in {"second_touch_rr", "lru"}:
        raise ValueError(f"{_CACHE_POLICY_ENV} must be second_touch_rr or lru")
    return value


def _validate_hot_lists(manifest: dict, rank: int) -> tuple[int, list[list[int]]]:
    if manifest.get("version") != 1:
        raise ValueError("Tiered expert manifest must use schema version 1")
    num_layers = manifest.get("num_layers")
    num_experts = manifest.get("num_experts")
    if num_layers != 48 or num_experts != 512:
        raise ValueError(
            "Tiered expert manifest must describe 48 layers and 512 experts"
        )
    try:
        rank_config = manifest["ranks"][str(rank)]
        hot_lists = rank_config["hot_experts_by_layer"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Tiered expert manifest has no TP rank {rank}") from error
    if not isinstance(hot_lists, list) or len(hot_lists) != num_layers:
        raise ValueError("Tiered expert manifest must contain one list per layer")
    for layer_id, hot_ids in enumerate(hot_lists):
        if not isinstance(hot_ids, list) or not hot_ids:
            raise ValueError(f"Layer {layer_id} has no hot expert list")
        if len(hot_ids) != len(set(hot_ids)):
            raise ValueError(f"Layer {layer_id} hot expert list contains duplicates")
        if any(not isinstance(expert, int) for expert in hot_ids):
            raise ValueError(f"Layer {layer_id} hot expert IDs must be integers")
        if min(hot_ids) < 0 or max(hot_ids) >= num_experts:
            raise ValueError(f"Layer {layer_id} hot expert ID is out of range")
    return num_experts, hot_lists


def _compact_expert_parameter(
    parameter: torch.nn.Parameter,
    hot_ids: list[int],
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if parameter.dtype != torch.uint8 or parameter.dim() != 3:
        raise TypeError("Tiered GGUF expert weights must be packed uint8 tensors")
    if parameter.shape[0] != num_experts or not parameter.is_contiguous():
        raise ValueError("Tiered GGUF expert master has an unexpected layout")
    if not getattr(parameter, "_vllm_is_uva_offloaded", False):
        raise RuntimeError("Tiered GGUF expert master was not UVA offloaded")
    cpu_master = getattr(parameter, "_vllm_uva_cpu_data", None)
    if cpu_master is None or not cpu_master.is_pinned():
        raise RuntimeError("Tiered GGUF expert master has no pinned CPU owner")

    device = parameter.device
    hot_ids_device = torch.tensor(hot_ids, dtype=torch.long, device=device)
    hot = parameter.index_select(0, hot_ids_device).contiguous()

    hot_set = set(hot_ids)
    cold_ids = [expert for expert in range(num_experts) if expert not in hot_set]
    cold_ids_cpu = torch.tensor(cold_ids, dtype=torch.long, device="cpu")
    cold_owner = allocate_uva_host_empty(
        (len(cold_ids), *parameter.shape[1:]), parameter.dtype
    )
    torch.index_select(cpu_master, 0, cold_ids_cpu, out=cold_owner)

    hot_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    cold_map = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    hot_map[hot_ids_device] = torch.arange(
        len(hot_ids), dtype=torch.int32, device=device
    )
    cold_ids_device = cold_ids_cpu.to(device=device)
    cold_map[cold_ids_device] = torch.arange(
        len(cold_ids), dtype=torch.int32, device=device
    )

    torch.cuda.synchronize(device)
    cold = get_accelerator_view_from_cpu_tensor(cold_owner)
    parameter.data = cold
    parameter._vllm_uva_cpu_data = cold_owner
    parameter._vllm_is_uva_offloaded = True
    return hot, hot_map, cold_map, hot.numel(), cold_owner.numel()


def materialize_hot_expert_cache(model: torch.nn.Module) -> None:
    manifest_path = os.environ.get(_MANIFEST_ENV)
    if not manifest_path:
        return
    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    rank = get_tensor_model_parallel_rank()
    num_experts, hot_lists = _validate_hot_lists(manifest, rank)
    cache_slots = _dynamic_cache_slots(rank)
    cache_policy = _cache_policy()
    async_cache = _async_cache_enabled() and cache_slots > 0
    if async_cache and cache_policy != "second_touch_rr":
        raise ValueError(
            f"{_CACHE_ASYNC_ENV}=1 is incompatible with "
            f"{_CACHE_POLICY_ENV}={cache_policy}"
        )
    physical_cache_slots = cache_slots + int(async_cache)

    from .fused_moe import GGUFMoEMethod, _tiered_iq_moe_hip

    layers = 0
    hot_bytes = 0
    cold_bytes = 0
    dynamic_cache_bytes = 0
    async_initialized = False
    for module in model.modules():
        if not isinstance(module, RoutedExperts) or not isinstance(
            getattr(module, "quant_method", None), GGUFMoEMethod
        ):
            continue
        match = _LAYER_PATTERN.search(module.layer_name)
        if match is None:
            continue
        layer_id = extract_layer_index(module.layer_name)
        if layer_id != int(match.group(1)) or not 0 <= layer_id < 48:
            continue
        hot_ids = hot_lists[layer_id]
        hot_w13, hot_map, cold_map, w13_hot, w13_cold = _compact_expert_parameter(
            module.w13_qweight, hot_ids, num_experts
        )
        hot_w2, w2_hot_map, w2_cold_map, w2_hot, w2_cold = _compact_expert_parameter(
            module.w2_qweight, hot_ids, num_experts
        )
        if not torch.equal(hot_map, w2_hot_map) or not torch.equal(
            cold_map, w2_cold_map
        ):
            raise RuntimeError("Tiered GGUF expert maps differ between projections")
        module.register_buffer("_gguf_hot_w13", hot_w13, persistent=False)
        module.register_buffer("_gguf_hot_w2", hot_w2, persistent=False)
        module.register_buffer("_gguf_global_to_hot", hot_map, persistent=False)
        module.register_buffer("_gguf_global_to_cold", cold_map, persistent=False)
        if cache_slots:
            device = hot_w13.device
            cache_w13 = torch.empty(
                (physical_cache_slots, *hot_w13.shape[1:]),
                dtype=torch.uint8,
                device=device,
            )
            cache_w2 = torch.empty(
                (physical_cache_slots, *hot_w2.shape[1:]),
                dtype=torch.uint8,
                device=device,
            )
            module.register_buffer("_gguf_cache_w13", cache_w13, persistent=False)
            module.register_buffer("_gguf_cache_w2", cache_w2, persistent=False)
            module.register_buffer(
                "_gguf_global_to_cache",
                torch.full((num_experts,), -1, dtype=torch.int32, device=device),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_tags",
                torch.full(
                    (physical_cache_slots,), -1, dtype=torch.int32, device=device
                ),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_clock",
                torch.zeros(
                    (physical_cache_slots + 1,),
                    dtype=torch.int32,
                    device=device,
                ),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_admission",
                torch.zeros((num_experts,), dtype=torch.int32, device=device),
                persistent=False,
            )
            # prepare calls, fills, routed hits, first-touch bypasses, evictions,
            # async schedules/fallbacks, and sync duplicate fills/served routes.
            module.register_buffer(
                "_gguf_cache_stats",
                torch.zeros((9,), dtype=torch.int32, device=device),
                persistent=False,
            )
            module.register_buffer(
                "_gguf_cache_pending",
                torch.zeros((7,), dtype=torch.int32, device=device),
                persistent=False,
            )
            module._gguf_cache_layer_id = layer_id
            module._gguf_cache_policy = cache_policy
            if cache_policy == "lru" and not hasattr(
                _tiered_iq_moe_hip(), "tiered_iq_moe_cache_lru_prepare"
            ):
                raise RuntimeError(
                    "The loaded tiered HIP extension does not support "
                    "the synchronous LRU expert cache"
                )
            if async_cache and not async_initialized:
                extension = _tiered_iq_moe_hip()
                if not all(
                    hasattr(extension, name)
                    for name in (
                        "tiered_iq_moe_cache_async_init",
                        "tiered_iq_moe_cache_async_prepare",
                        "tiered_iq_moe_cache_async_commit",
                    )
                ):
                    raise RuntimeError(
                        "The loaded tiered HIP extension does not support "
                        "asynchronous expert-cache fills"
                    )
                with torch.cuda.device(device):
                    extension.tiered_iq_moe_cache_async_init()
                async_initialized = True
            dynamic_cache_bytes += cache_w13.numel() + cache_w2.numel()
        layers += 1
        hot_bytes += w13_hot + w2_hot
        cold_bytes += w13_cold + w2_cold

    if layers != 48:
        raise RuntimeError(
            f"Tiered GGUF cache found {layers} target layers, expected 48"
        )
    host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
    if host_empty_cache is not None:
        host_empty_cache()
    logger.warning(
        "Tiered GGUF experts ready on TP rank %d: layers=%d hot=%.3f GiB "
        "cold_UVA=%.3f GiB dynamic_cache=%.3f GiB cache_slots=%d "
        "physical_cache_slots=%d cache_policy=%s async_cache=%s manifest=%s",
        rank,
        layers,
        hot_bytes / 1024**3,
        cold_bytes / 1024**3,
        dynamic_cache_bytes / 1024**3,
        cache_slots,
        physical_cache_slots,
        cache_policy,
        async_cache,
        path,
    )
