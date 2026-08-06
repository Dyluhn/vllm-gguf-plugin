# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Protocol

import torch


class GGUFLinearLayout(Protocol):
    """A layout transform shared by GGUF linear inputs and weights."""

    def input_to_gguf(self, x: torch.Tensor) -> torch.Tensor: ...

    def weight_to_vllm(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        head_dim: int | None = None,
    ) -> torch.Tensor: ...

    def shard_weight(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        logical_size: int,
        block_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class GGUFHeadTilingLayout:
    """Describe grouped heads stored as head-major tiles by GGML."""

    heads_per_group: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.heads_per_group <= 1:
            raise ValueError("heads_per_group must be greater than one")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

    def input_to_gguf(self, x: torch.Tensor) -> torch.Tensor:
        num_heads, remainder = divmod(x.shape[-1], self.head_dim)
        if remainder or num_heads % self.heads_per_group:
            raise ValueError(
                "Cannot reorder linear input shape "
                f"{tuple(x.shape)} with heads_per_group={self.heads_per_group}, "
                f"head_dim={self.head_dim}"
            )
        num_groups = num_heads // self.heads_per_group
        shape = (*x.shape[:-1], num_groups, self.heads_per_group, self.head_dim)
        return x.reshape(shape).transpose(-3, -2).reshape(x.shape).contiguous()

    def weight_to_vllm(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        head_dim: int | None = None,
    ) -> torch.Tensor:
        """Restore a GGML head-tiled dimension to the vLLM head order."""
        if dim < 0:
            dim += weight.ndim
        if not 0 <= dim < weight.ndim:
            raise ValueError(
                f"Invalid dimension {dim} for shape {tuple(weight.shape)}"
            )

        head_dim = self.head_dim if head_dim is None else head_dim
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        num_heads, remainder = divmod(weight.shape[dim], head_dim)
        num_groups, group_remainder = divmod(num_heads, self.heads_per_group)
        if remainder or group_remainder:
            raise ValueError(
                "Cannot restore GGML weight shape "
                f"{tuple(weight.shape)} along dim={dim} with "
                f"heads_per_group={self.heads_per_group}, head_dim={head_dim}"
            )

        shape = list(weight.shape)
        tiled_shape = (
            *shape[:dim],
            self.heads_per_group,
            num_groups,
            head_dim,
            *shape[dim + 1 :],
        )
        return (
            weight.reshape(tiled_shape)
            .transpose(dim, dim + 1)
            .reshape(shape)
            .contiguous()
        )

    def shard_weight(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        logical_size: int,
        block_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        """Select this rank's groups from every stored head tile.

        The weight may contain scalar values or packed GGML blocks. Offsets are
        computed in logical elements and converted to packed bytes, so
        quantized weights stay quantized while being sharded.
        """
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"Invalid TP rank {tp_rank} for TP size {tp_size}")
        if tp_size == 1:
            return weight
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        num_heads, remainder = divmod(logical_size, self.head_dim)
        if remainder or num_heads % self.heads_per_group:
            raise ValueError(
                f"Cannot shard logical input size {logical_size} with "
                f"heads_per_group={self.heads_per_group}, "
                f"head_dim={self.head_dim}"
            )
        num_groups = num_heads // self.heads_per_group
        local_groups, remainder = divmod(num_groups, tp_size)
        if remainder:
            raise ValueError(
                f"Cannot divide {num_groups} head groups across TP size {tp_size}"
            )

        logical_blocks, remainder = divmod(logical_size, block_size)
        if remainder:
            raise ValueError(
                f"Logical input size {logical_size} is not aligned to GGML "
                f"block size {block_size}"
            )
        packed_block_size, remainder = divmod(weight.shape[dim], logical_blocks)
        if remainder:
            raise ValueError(
                f"Packed weight dimension {weight.shape[dim]} does not match "
                f"logical input size {logical_size} and block size {block_size}"
            )

        group_span = local_groups * self.head_dim
        if group_span % block_size:
            raise ValueError(
                f"TP size {tp_size} splits a stored head tile at {group_span} "
                f"logical elements, which is not aligned to GGML block size "
                f"{block_size}"
            )

        shards = []
        for head in range(self.heads_per_group):
            logical_offset = head * num_groups * self.head_dim + tp_rank * group_span
            if logical_offset % block_size:
                raise ValueError(
                    f"TP rank {tp_rank} starts at unaligned logical offset "
                    f"{logical_offset} for GGML block size {block_size}"
                )
            block_offset = logical_offset // block_size
            block_count = group_span // block_size
            shards.append(
                weight.narrow(
                    dim,
                    block_offset * packed_block_size,
                    block_count * packed_block_size,
                )
            )
        return torch.cat(shards, dim=dim).contiguous()
