# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

import vllm_gguf_plugin.weight_utils as weight_utils_module
import vllm_gguf_plugin.weights_adapter.base as base_module
import vllm_gguf_plugin.weights_adapter.gemma3 as gemma3_module
import vllm_gguf_plugin.weights_adapter.olmoe as olmoe_module
from vllm_gguf_plugin.gguf_files import (
    GGUFModelFiles,
    resolve_gguf_model_files,
)
from vllm_gguf_plugin.weight_utils import (
    get_gguf_shard_files,
    gguf_quant_weights_iterator_multi,
)
from vllm_gguf_plugin.weights_adapter import (
    BaseGGUFWeightsAdapter,
    Gemma3GGUFAdapter,
    OLMoEGGUFAdapter,
    TransformersGGUFWeightsAdapter,
    get_weights_adapter,
)
from vllm_gguf_plugin.weights_adapter.base import GGUFWeight


class _TestAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, config) -> bool:
        del config
        return True

    def build_name_map(self, files, model_config) -> dict[str, str]:
        del files
        assert model_config.hf_config.tie_word_embeddings is True
        return {
            "token_embd.weight": "model.embed_tokens.weight",
            "output_norm.weight": "model.norm.weight",
        }

    def select_tensor_names(self, files, model_config) -> Iterable[str]:
        del files, model_config
        return ("token_embd.weight", "output_norm.weight")

    def transform_weights(
        self,
        weights: Iterable[GGUFWeight],
        model_config,
    ) -> Iterable[GGUFWeight]:
        del model_config
        for name, weight in weights:
            yield f"transformed.{name}", weight


def test_resolve_gguf_model_files_groups_shards_and_mmproj(tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    mm_proj = tmp_path / "mmproj.gguf"
    for path in (first, second, mm_proj):
        path.write_bytes(b"GGUF")

    files = resolve_gguf_model_files(str(first))

    assert files.backbone == (str(first), str(second))
    assert files.mm_proj == str(mm_proj)
    assert files.all_files == (str(first), str(second), str(mm_proj))


def test_get_gguf_shard_files_rejects_incomplete_set(tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    first.write_bytes(b"GGUF")

    with pytest.raises(FileNotFoundError, match="Missing 1 of 2"):
        get_gguf_shard_files(str(first))


def test_common_prepare_builds_plan_from_all_file_roles(monkeypatch):
    seen_files = []

    def fake_unquantized(files):
        seen_files.extend(files)
        return ["token_embd.weight", "output_norm.weight", "ignored.bias"]

    monkeypatch.setattr(
        base_module,
        "get_gguf_unquantized_params",
        fake_unquantized,
    )
    monkeypatch.setattr(
        base_module,
        "get_gguf_tensor_names",
        lambda files: {"token_embd.weight"},
    )

    files = GGUFModelFiles(("backbone-1.gguf", "backbone-2.gguf"), "mmproj.gguf")
    model_config = SimpleNamespace(hf_config=PretrainedConfig())

    plan = _TestAdapter().prepare(files, model_config)

    assert seen_files == list(files.all_files)
    assert plan.files is files
    assert plan.name_map["token_embd.weight"] == "model.embed_tokens.weight"
    assert plan.selected_tensors == frozenset(
        {"token_embd.weight", "output_norm.weight"}
    )
    assert plan.unquantized_modules == (
        "model.embed_tokens",
        "model.norm",
    )
    assert model_config.hf_config.tie_word_embeddings is True


def test_common_iter_weights_applies_adapter_transform(monkeypatch):
    tensor = torch.ones(2)
    captured = {}

    def fake_iterator(files, name_map, **kwargs):
        captured.update(files=files, name_map=name_map, **kwargs)
        return iter([("model.norm.weight", tensor)])

    monkeypatch.setattr(
        base_module,
        "gguf_quant_weights_iterator_multi",
        fake_iterator,
    )
    files = GGUFModelFiles(("model.gguf",))
    plan = base_module.GGUFLoadPlan(
        files,
        {},
        (),
        selected_tensors=frozenset({"raw.weight"}),
    )

    weights = list(
        _TestAdapter().iter_weights(
            plan,
            SimpleNamespace(hf_config=PretrainedConfig()),
        )
    )

    assert weights == [("transformed.model.norm.weight", tensor)]
    assert captured["selected_tensors"] == frozenset({"raw.weight"})


def test_weight_iterator_selects_and_preserves_quantized_tensors(monkeypatch):
    keep = SimpleNamespace(
        name="keep.weight",
        tensor_type=gguf.GGMLQuantizationType.Q4_0,
        data=np.zeros(18, dtype=np.uint8),
    )
    skip = SimpleNamespace(
        name="skip.weight",
        tensor_type=gguf.GGMLQuantizationType.Q4_0,
        data=np.zeros(18, dtype=np.uint8),
    )
    reader = SimpleNamespace(tensors=[keep, skip], byte_order="S")
    monkeypatch.setattr(weight_utils_module.gguf, "GGUFReader", lambda path: reader)
    weights = list(
        gguf_quant_weights_iterator_multi(
            ["model.gguf"],
            {"keep.weight": "model.keep.weight"},
            selected_tensors={"keep.weight"},
        )
    )

    assert [name for name, _ in weights] == [
        "model.keep.qweight_type",
        "model.keep.qweight",
    ]
    assert weights[0][1].item() == gguf.GGMLQuantizationType.Q4_0
    assert torch.equal(weights[1][1], torch.from_numpy(keep.data))


def test_gemma3_config_patch_uses_resolved_mmproj(monkeypatch):
    captured = {}

    def fake_patch(model, config, mmproj_path=None):
        captured.update(model=model, config=config, mmproj_path=mmproj_path)
        return config

    monkeypatch.setattr(gemma3_module, "maybe_patch_hf_config_from_gguf", fake_patch)
    config = PretrainedConfig()
    files = GGUFModelFiles(("model.gguf",), "/other/mmproj.gguf")

    assert Gemma3GGUFAdapter().patch_hf_config(files, config) is config
    assert captured == {
        "model": "model.gguf",
        "config": config,
        "mmproj_path": "/other/mmproj.gguf",
    }


def test_gemma3_name_map_covers_backbone_and_mmproj(monkeypatch):
    monkeypatch.setattr(
        gemma3_module,
        "get_gguf_tensor_names",
        lambda files: {
            "blk.0.attn_q.weight",
            "v.blk.0.attn_q.weight",
            "mm.input_projection.weight",
        },
    )
    files = GGUFModelFiles(("model.gguf",), "mmproj.gguf")

    name_map = Gemma3GGUFAdapter().build_name_map(files, SimpleNamespace())

    assert name_map["blk.0.attn_q.weight"] == (
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )
    assert name_map["v.blk.0.attn_q.weight"] == (
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    )
    assert name_map["mm.input_projection.weight"] == (
        "multi_modal_projector.mm_input_projection_weight"
    )


def test_olmoe_name_map_uses_manual_mapper(monkeypatch):
    monkeypatch.setattr(
        olmoe_module,
        "get_gguf_tensor_names",
        lambda files: {
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        },
    )
    files = GGUFModelFiles(("model.gguf",))

    name_map = OLMoEGGUFAdapter().build_name_map(files, SimpleNamespace())

    assert name_map["token_embd.weight"] == "model.embed_tokens.weight"
    assert name_map["blk.0.attn_q.weight"] == ("model.layers.0.self_attn.q_proj.weight")
    assert name_map["blk.0.ffn_gate_exps.weight"] == (
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    )
    assert name_map["blk.0.ffn_up_exps.weight"] == (
        "model.layers.0.mlp.experts.0.up_proj.weight"
    )
    assert name_map["blk.0.ffn_down_exps.weight"] == (
        "model.layers.0.mlp.experts.0.down_proj.weight"
    )


def test_olmoe_transform_splits_expert_dimension():
    gate = torch.arange(24).reshape(2, 3, 4)
    qweight_type = torch.tensor(2)
    weights = list(
        OLMoEGGUFAdapter().transform_weights(
            [
                ("model.layers.0.mlp.experts.0.gate_proj.qweight", gate),
                (
                    "model.layers.0.mlp.experts.0.gate_proj.qweight_type",
                    qweight_type,
                ),
            ],
            SimpleNamespace(),
        )
    )

    assert [name for name, _ in weights] == [
        "model.layers.0.mlp.experts.0.gate_proj.qweight",
        "model.layers.0.mlp.experts.1.gate_proj.qweight",
        "model.layers.0.mlp.experts.0.gate_proj.qweight_type",
    ]
    assert torch.equal(weights[0][1], gate[0])
    assert torch.equal(weights[1][1], gate[1])
    assert weights[2][1] is qweight_type


def test_adapter_factory_uses_specialized_adapter_before_fallback():
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="gemma3_text")),
        Gemma3GGUFAdapter,
    )
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="olmoe")),
        OLMoEGGUFAdapter,
    )
    assert isinstance(
        get_weights_adapter(PretrainedConfig(model_type="llama")),
        TransformersGGUFWeightsAdapter,
    )
