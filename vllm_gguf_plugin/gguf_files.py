# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

from .gguf_utils import detect_gguf_multimodal
from .weight_utils import get_gguf_shard_files


@dataclass(frozen=True, slots=True)
class GGUFModelFiles:
    """Resolved GGUF files grouped by their role in one model."""

    backbone: tuple[str, ...]
    mm_proj: str | None = None

    def __post_init__(self) -> None:
        if not self.backbone:
            raise ValueError("GGUFModelFiles requires at least one backbone file")

    @property
    def primary_backbone(self) -> str:
        return self.backbone[0]

    @property
    def all_files(self) -> tuple[str, ...]:
        if self.mm_proj is None:
            return self.backbone
        return (*self.backbone, self.mm_proj)


def resolve_gguf_model_files(
    model_path: str,
    mm_proj_path: str | None = None,
) -> GGUFModelFiles:
    """Resolve backbone shards and an optional sibling multimodal projector."""
    if mm_proj_path is None:
        detected_mm_proj = detect_gguf_multimodal(model_path)
        mm_proj_path = str(detected_mm_proj) if detected_mm_proj is not None else None
    return GGUFModelFiles(
        backbone=tuple(get_gguf_shard_files(model_path)),
        mm_proj=mm_proj_path,
    )
