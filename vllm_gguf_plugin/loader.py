# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import cast

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from .gguf_files import GGUFModelFiles, resolve_gguf_model_files
from .gguf_utils import get_remote_gguf_repo_id, resolve_explicit_mm_proj
from .quantization import GGUFConfig, recursive_replace_vocab_modules
from .weight_utils import (
    download_gguf,
    download_mmproj,
    resolve_local_gguf,
)
from .weights_adapter import get_weights_adapter

logger = init_logger(__name__)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config or {}
        unknown_options = set(extra_config) - {"mm_proj"}
        if unknown_options:
            raise ValueError(
                "Unsupported GGUF model loader extra config options: "
                f"{sorted(unknown_options)}"
            )
        self._mm_proj_reference = extra_config.get("mm_proj")

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model_weights or model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # local_dir:quant_type (e.g. /path/to/gguf-dir:Q8_0)
        if ":" in model_name_or_path:
            local_dir, quant_type = model_name_or_path.rsplit(":", 1)
            if os.path.isdir(local_dir):
                return resolve_local_gguf(local_dir, quant_type)
            # remote repo_id:quant_type
            return download_gguf(
                local_dir,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, <local_dir>:<quant_type>, "
            "<repo_id>/<filename>.gguf, or <repo_id>:<quant_type>)"
        )

    def _prepare_model_files(self, model_config: ModelConfig) -> GGUFModelFiles:
        model_reference = model_config.model_weights or model_config.model
        model_path = self._prepare_weights(model_config)
        files = resolve_gguf_model_files(model_path)
        if self._mm_proj_reference is not None:
            mm_proj_path = resolve_explicit_mm_proj(
                self._mm_proj_reference,
                model_path,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
            )
            return resolve_gguf_model_files(model_path, mm_proj_path)

        if (
            files.mm_proj is None
            and getattr(model_config.hf_config, "vision_config", None) is not None
        ):
            repo_id = get_remote_gguf_repo_id(model_reference)
            if repo_id is not None:
                mm_proj_path = download_mmproj(
                    repo_id,
                    cache_dir=self.load_config.download_dir,
                    revision=model_config.revision,
                )
                if mm_proj_path is not None:
                    return resolve_gguf_model_files(model_path, mm_proj_path)

        return files

    def _prepare_adapter(self, model_config: ModelConfig):
        files = self._prepare_model_files(model_config)
        adapter = get_weights_adapter(model_config.hf_config)
        plan = adapter.prepare(files, model_config)
        return adapter, plan

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_model_files(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        adapter, plan = self._prepare_adapter(model_config)
        model.load_weights(adapter.iter_weights(plan, model_config))

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        adapter, plan = self._prepare_adapter(model_config)
        logger.debug("GGUF unquantized modules: %s", plan.unquantized_modules)
        vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(plan.unquantized_modules)
        vllm_config.quant_config.register_linear_layouts(
            plan.linear_layouts,
            prefix=prefix,
        )

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                recursive_replace_vocab_modules(
                    model,
                    vllm_config.quant_config,
                    prefix=prefix,
                )
            model.load_weights(
                adapter.iter_weights(plan, model_config),
            )
            process_weights_after_loading(model, model_config, target_device)
        return model
