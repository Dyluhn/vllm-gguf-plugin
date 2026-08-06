# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import gguf
import pytest
import torch
import torch.nn as nn
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_module
from transformers import PretrainedConfig
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

import vllm_gguf_plugin.weights_adapter.qwen3_5 as qwen_module
from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.quantization.config import GGUFConfig
from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod
from vllm_gguf_plugin.quantization.params import GGUFUninitializedParameter
from vllm_gguf_plugin.quantization.vocal_embeds import GGUFEmbeddingMethod
from vllm_gguf_plugin.weight_utils import split_stacked_experts
from vllm_gguf_plugin.weights_adapter import (
    Qwen35GGUFAdapter,
    Qwen35MtpGGUFAdapter,
    get_weights_adapter,
)
from vllm_gguf_plugin.weights_adapter.base import GGUFLoadPlan
from vllm_gguf_plugin.weights_adapter.qwen3_5 import (
    build_qwen35_mtp_mapper,
    build_qwen35_text_mapper,
    build_qwen35_vision_mapper,
)


@pytest.mark.parametrize(
    ("model_type", "adapter_type"),
    [
        ("qwen3_5", Qwen35GGUFAdapter),
        ("qwen3_5_text", Qwen35GGUFAdapter),
        ("qwen3_5_moe", Qwen35GGUFAdapter),
        ("qwen3_5_moe_text", Qwen35GGUFAdapter),
        ("qwen3_5_mtp", Qwen35MtpGGUFAdapter),
        ("qwen3_5_moe_mtp", Qwen35MtpGGUFAdapter),
    ],
)
def test_qwen35_adapter_registration(model_type, adapter_type):
    config = PretrainedConfig(model_type=model_type)
    assert isinstance(get_weights_adapter(config), adapter_type)


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "gemma3"])
def test_qwen35_adapter_does_not_match_other_models(model_type):
    config = PretrainedConfig(model_type=model_type)
    assert not Qwen35GGUFAdapter.matches(config)
    assert not Qwen35MtpGGUFAdapter.matches(config)


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ("qwen3_5_text", "Qwen3_5ForConditionalGeneration"),
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        ("qwen3_5_moe_text", "Qwen3_5MoeForConditionalGeneration"),
    ],
)
def test_patch_config_enforces_qwen_architecture(monkeypatch, model_type, architecture):
    config = PretrainedConfig(
        model_type=model_type,
        architectures=["Qwen3_5ForCausalLM"],
    )
    monkeypatch.setattr(
        qwen_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config, mmproj_path=None: config,
    )

    patched = Qwen35GGUFAdapter().patch_hf_config(
        GGUFModelFiles(("model.gguf",)),
        config,
    )

    assert patched.architectures == [architecture]


def test_patch_config_upgrades_text_model_when_mmproj_is_separate(monkeypatch):
    config = PretrainedConfig(model_type="qwen3_5_text")
    monkeypatch.setattr(
        qwen_module,
        "maybe_patch_hf_config_from_gguf",
        lambda model, config, mmproj_path=None: config,
    )

    patched = Qwen35GGUFAdapter().patch_hf_config(
        GGUFModelFiles(("model.gguf",), "mmproj.gguf"),
        config,
    )

    assert patched.model_type == "qwen3_5"
    assert patched.vision_config is not None
    assert patched.architectures == ["Qwen3_5ForConditionalGeneration"]


class TestTextMapper:
    def _map(self, name, is_multimodal=False, is_moe=False):
        mapper = build_qwen35_text_mapper(is_multimodal, is_moe)
        return mapper.apply_list([name])[0]

    @pytest.mark.parametrize(
        ("gguf_name", "hf_name"),
        [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
            ("blk.3.attn_q.weight", "model.layers.3.self_attn.q_proj.weight"),
            ("blk.3.attn_q_norm.weight", "model.layers.3.self_attn.q_norm.weight"),
            ("blk.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
            (
                "blk.3.post_attention_norm.weight",
                "model.layers.3.post_attention_layernorm.weight",
            ),
            (
                "blk.3.attn_qkv.weight",
                "model.layers.3.linear_attn.in_proj_qkv.weight",
            ),
            ("blk.3.attn_gate.weight", "model.layers.3.linear_attn.in_proj_z.weight"),
            ("blk.3.ssm_alpha.weight", "model.layers.3.linear_attn.in_proj_a.weight"),
            ("blk.3.ssm_beta.weight", "model.layers.3.linear_attn.in_proj_b.weight"),
            ("blk.3.ssm_out.weight", "model.layers.3.linear_attn.out_proj.weight"),
            ("blk.3.ssm_norm.weight", "model.layers.3.linear_attn.norm.weight"),
            ("blk.3.ssm_conv1d.weight", "model.layers.3.linear_attn.conv1d.weight"),
            ("blk.3.ssm_a.weight", "model.layers.3.linear_attn.A_log"),
            ("blk.3.ssm_dt.bias", "model.layers.3.linear_attn.dt_bias"),
        ],
    )
    def test_maps_text_tensors(self, gguf_name, hf_name):
        assert self._map(gguf_name) == hf_name

    def test_multimodal_uses_language_model_prefix(self):
        assert (
            self._map("blk.3.attn_q.weight", is_multimodal=True)
            == "model.language_model.layers.3.self_attn.q_proj.weight"
        )
        assert self._map("output.weight", is_multimodal=True) == "lm_head.weight"

    @pytest.mark.parametrize(
        ("gguf_name", "hf_name"),
        [
            ("blk.1.ffn_gate.weight", "model.layers.1.mlp.gate_proj.weight"),
            ("blk.1.ffn_up.weight", "model.layers.1.mlp.up_proj.weight"),
            ("blk.1.ffn_down.weight", "model.layers.1.mlp.down_proj.weight"),
        ],
    )
    def test_maps_dense_mlp(self, gguf_name, hf_name):
        assert self._map(gguf_name) == hf_name

    @pytest.mark.parametrize(
        ("gguf_name", "hf_name"),
        [
            ("blk.1.ffn_gate_inp.weight", "model.layers.1.mlp.gate.weight"),
            (
                "blk.1.ffn_gate_inp_shexp.weight",
                "model.layers.1.mlp.shared_expert_gate.weight",
            ),
            (
                "blk.1.ffn_gate_exps.weight",
                "model.layers.1.mlp.experts.0.gate_proj.weight",
            ),
            ("blk.1.ffn_up_exps.weight", "model.layers.1.mlp.experts.0.up_proj.weight"),
            (
                "blk.1.ffn_down_exps.weight",
                "model.layers.1.mlp.experts.0.down_proj.weight",
            ),
            (
                "blk.1.ffn_up_shexp.weight",
                "model.layers.1.mlp.shared_expert.up_proj.weight",
            ),
        ],
    )
    def test_maps_moe_mlp(self, gguf_name, hf_name):
        assert self._map(gguf_name, is_moe=True) == hf_name


def test_every_qwen35_arch_tensor_has_a_mapping_rule():
    known_unmapped = {"ffn_gate_up_exps"}
    for arch_name, is_moe in (("QWEN35", False), ("QWEN35MOE", True)):
        arch = getattr(gguf.MODEL_ARCH, arch_name)
        mapper = build_qwen35_text_mapper(False, is_moe)
        base_names = {
            base for _, base in gguf.get_tensor_name_map(arch, 1).mapping.values()
        }
        unmapped = sorted(
            base
            for base in base_names
            if base.rsplit(".", 1)[-1] not in known_unmapped
            and all(
                mapper.apply_list([f"{base}.{suffix}"])[0] == f"{base}.{suffix}"
                for suffix in ("weight", "bias")
            )
        )
        assert unmapped == []


@pytest.mark.parametrize(
    ("gguf_name", "hf_name"),
    [
        ("v.patch_embd.weight", "model.visual.patch_embed.proj.weight"),
        ("v.position_embd.weight", "model.visual.pos_embed.weight"),
        ("v.blk.2.attn_qkv.weight", "model.visual.blocks.2.attn.qkv.weight"),
        ("v.blk.2.attn_out.bias", "model.visual.blocks.2.attn.proj.bias"),
        ("v.blk.2.ffn_up.weight", "model.visual.blocks.2.mlp.linear_fc1.weight"),
        ("v.blk.2.ffn_down.bias", "model.visual.blocks.2.mlp.linear_fc2.bias"),
        ("v.blk.2.ln1.weight", "model.visual.blocks.2.norm1.weight"),
        ("v.blk.2.ln2.bias", "model.visual.blocks.2.norm2.bias"),
        ("mm.0.weight", "model.visual.merger.linear_fc1.weight"),
        ("mm.2.bias", "model.visual.merger.linear_fc2.bias"),
        ("v.post_ln.weight", "model.visual.merger.norm.weight"),
    ],
)
def test_maps_vision_tensors(gguf_name, hf_name):
    mapper = build_qwen35_vision_mapper()
    assert mapper.apply_list([gguf_name])[0] == hf_name


def test_build_name_map_combines_backbone_and_mmproj(monkeypatch):
    names = {
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "v.blk.0.attn_qkv.weight",
        "mm.0.weight",
        "blk.1.nextn.eh_proj.weight",
    }
    monkeypatch.setattr(qwen_module, "get_gguf_tensor_names", lambda files: names)
    config = PretrainedConfig(model_type="qwen3_5")
    model_config = SimpleNamespace(hf_config=config)
    files = GGUFModelFiles(("model.gguf",), "mmproj.gguf")

    name_map = Qwen35GGUFAdapter().build_name_map(files, model_config)

    assert name_map["token_embd.weight"] == ("model.language_model.embed_tokens.weight")
    assert name_map["v.blk.0.attn_qkv.weight"] == (
        "model.visual.blocks.0.attn.qkv.weight"
    )
    assert name_map["mm.0.weight"] == "model.visual.merger.linear_fc1.weight"
    assert "blk.1.nextn.eh_proj.weight" not in name_map


def test_qwen_gdn_declares_out_proj_input_layout_before_model_init():
    config = PretrainedConfig(
        model_type="qwen3_5",
        linear_num_key_heads=2,
        linear_num_value_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    model_config = SimpleNamespace(hf_config=config)
    module_name = "model.layers.0.linear_attn.out_proj"
    name_map = {"blk.0.ssm_out.weight": f"{module_name}.weight"}

    layouts = Qwen35GGUFAdapter().get_linear_layouts(
        GGUFModelFiles(("model.gguf",)),
        model_config,
        name_map,
    )

    assert layouts == {
        module_name: GGUFHeadTilingLayout(
            heads_per_group=4,
            head_dim=128,
        )
    }

    quant_config = GGUFConfig()
    quant_config.register_linear_layouts(layouts)
    quant_method = quant_config.get_quant_method(
        object.__new__(LinearBase),
        module_name,
    )

    assert type(quant_method) is GGUFLinearMethod
    assert quant_method.layout is layouts[module_name]
    other_method = quant_config.get_quant_method(
        object.__new__(LinearBase),
        "model.layers.0.mlp.down_proj",
    )
    assert type(other_method) is GGUFLinearMethod

    unquantized_config = GGUFConfig(unquantized_modules=[module_name])
    unquantized_config.register_linear_layouts(layouts)
    unquantized_method = unquantized_config.get_quant_method(
        object.__new__(LinearBase),
        module_name,
    )
    assert isinstance(unquantized_method, UnquantizedLinearMethod)

    prefixed_config = GGUFConfig()
    prefixed_config.register_linear_layouts(
        layouts,
        prefix="draft",
    )
    prefixed_method = prefixed_config.get_quant_method(
        object.__new__(LinearBase),
        f"draft.{module_name}",
    )
    assert type(prefixed_method) is GGUFLinearMethod
    assert prefixed_method.layout is layouts[module_name]


def test_configure_model_enables_packed_token_embedding(monkeypatch):
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_world_size", lambda: 1)
    embedding = VocabParallelEmbedding(
        num_embeddings=10,
        embedding_dim=4,
        org_num_embeddings=10,
        padding_size=8,
    )
    model = nn.Module()
    model.embed_tokens = embedding
    files = GGUFModelFiles(("model.gguf",))
    plan = GGUFLoadPlan(
        files=files,
        name_map={"token_embd.weight": "model.embed_tokens.weight"},
        unquantized_modules=(),
    )
    quant_config = GGUFConfig()

    Qwen35GGUFAdapter().configure_model(
        model,
        plan,
        SimpleNamespace(hf_config=PretrainedConfig()),
        quant_config,
    )

    assert embedding.weight is None
    assert isinstance(embedding.quant_method, GGUFEmbeddingMethod)
    assert isinstance(embedding.qweight, GGUFUninitializedParameter)
    assert isinstance(embedding.qweight_type, GGUFUninitializedParameter)


def test_runtime_input_reorder_matches_logical_weight_reorder():
    adapter = Qwen35GGUFAdapter()
    stored_weight = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    text_config = PretrainedConfig(
        linear_num_key_heads=2,
        linear_key_head_dim=1,
    )
    layout = GGUFHeadTilingLayout(
        heads_per_group=4,
        head_dim=1,
    )
    logical_weight = adapter._restore_gdn_weight(
        "model.layers.0.linear_attn.out_proj.weight",
        stored_weight,
        text_config,
        layout,
    )
    logical_input = torch.randn(3, 8)
    packed_input = layout.input_to_gguf(logical_input)

    assert torch.allclose(
        logical_input @ logical_weight.T,
        packed_input @ stored_weight.T,
    )


def test_gdn_row_reorder_operates_on_packed_rows():
    adapter = Qwen35GGUFAdapter()
    text_config = PretrainedConfig(
        linear_num_key_heads=2,
        linear_key_head_dim=1,
    )
    layout = GGUFHeadTilingLayout(heads_per_group=2, head_dim=1)
    weight = torch.arange(16).reshape(8, 2)

    reordered = adapter._restore_gdn_weight(
        "model.layers.0.linear_attn.in_proj_qkv.qweight",
        weight,
        text_config,
        layout,
    )

    assert torch.equal(reordered[:4], weight[:4])
    assert torch.equal(reordered[4:], weight[[4, 6, 5, 7]])


def test_transform_keeps_quantized_embeddings_and_lm_head():
    config = SimpleNamespace(
        hf_config=PretrainedConfig(model_type="qwen3_5"),
    )
    weights = [
        ("model.embed_tokens.qweight_type", torch.tensor(2)),
        ("model.embed_tokens.qweight", torch.ones(3, 4, dtype=torch.uint8)),
        ("lm_head.qweight_type", torch.tensor(2)),
        ("lm_head.qweight", torch.ones(3, 4, dtype=torch.uint8)),
    ]

    transformed = list(Qwen35GGUFAdapter().transform_weights(weights, config))

    assert [name for name, _ in transformed] == [name for name, _ in weights]
    assert all(weight.dtype != torch.float32 for _, weight in transformed)


def test_split_stacked_expert_weight():
    name = "model.layers.0.mlp.experts.0.gate_proj.qweight"
    output = dict(split_stacked_experts([(name, torch.zeros(3, 4, 2))]))
    assert list(output) == [
        "model.layers.0.mlp.experts.0.gate_proj.qweight",
        "model.layers.0.mlp.experts.1.gate_proj.qweight",
        "model.layers.0.mlp.experts.2.gate_proj.qweight",
    ]


def test_transform_stacks_temporal_patch_embed_slices():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vision_config=SimpleNamespace(temporal_patch_size=2),
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    name = "model.visual.patch_embed.proj.weight"
    first = torch.ones(8, 3, 14, 14)
    second = torch.zeros(8, 3, 14, 14)
    weights = [(f"{name}.1", second), (name, first)]

    output = dict(Qwen35GGUFAdapter().transform_weights(weights, config))

    assert output[name].shape == (8, 3, 2, 14, 14)
    assert torch.equal(output[name][:, :, 0], first)
    assert torch.equal(output[name][:, :, 1], second)


_MTP_MOE_BLOCK = {
    "nextn.eh_proj.weight": "mtp.fc.weight",
    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
    "attn_norm.weight": "mtp.layers.0.input_layernorm.weight",
    "post_attention_norm.weight": "mtp.layers.0.post_attention_layernorm.weight",
    "attn_q.weight": "mtp.layers.0.self_attn.q_proj.weight",
    "attn_k.weight": "mtp.layers.0.self_attn.k_proj.weight",
    "attn_v.weight": "mtp.layers.0.self_attn.v_proj.weight",
    "attn_output.weight": "mtp.layers.0.self_attn.o_proj.weight",
    "attn_q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
    "attn_k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mtp.layers.0.mlp.gate.weight",
    "ffn_gate_exps.weight": "mtp.layers.0.mlp.experts.0.gate_proj.weight",
    "ffn_up_exps.weight": "mtp.layers.0.mlp.experts.0.up_proj.weight",
    "ffn_down_exps.weight": "mtp.layers.0.mlp.experts.0.down_proj.weight",
    "ffn_gate_shexp.weight": "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "ffn_gate_inp_shexp.weight": "mtp.layers.0.mlp.shared_expert_gate.weight",
}


def test_mtp_mapper_covers_complete_moe_block():
    mapper = build_qwen35_mtp_mapper(40, is_moe=True)
    names = [f"blk.40.{suffix}" for suffix in _MTP_MOE_BLOCK]
    assert mapper.apply_list(names) == list(_MTP_MOE_BLOCK.values())


def test_find_nextn_block_index_across_shards(monkeypatch):
    readers = [
        SimpleNamespace(tensors=[SimpleNamespace(name="blk.0.attn_q.weight")]),
        SimpleNamespace(tensors=[SimpleNamespace(name="blk.40.nextn.eh_proj.weight")]),
    ]
    monkeypatch.setattr(qwen_module.gguf, "GGUFReader", lambda path: readers.pop(0))

    assert qwen_module._find_nextn_block_index(["part-1.gguf", "part-2.gguf"]) == 40


def test_mtp_build_name_map_rejects_gguf_without_nextn(monkeypatch):
    monkeypatch.setattr(qwen_module, "_find_nextn_block_index", lambda files: None)
    model_config = SimpleNamespace(hf_config=PretrainedConfig(model_type="qwen3_5_mtp"))

    with pytest.raises(RuntimeError, match="No MTP/nextn block"):
        Qwen35MtpGGUFAdapter().build_name_map(
            GGUFModelFiles(("model.gguf",)), model_config
        )


def test_mtp_build_name_map_selects_only_nextn_block(monkeypatch):
    names = {
        "blk.39.attn_q.weight",
        "blk.40.nextn.eh_proj.weight",
        "blk.40.attn_q.weight",
    }
    monkeypatch.setattr(qwen_module, "get_gguf_tensor_names", lambda files: names)
    monkeypatch.setattr(qwen_module, "_find_nextn_block_index", lambda files: 40)
    model_config = SimpleNamespace(
        hf_config=PretrainedConfig(model_type="qwen3_5_moe_mtp")
    )

    name_map = Qwen35MtpGGUFAdapter().build_name_map(
        GGUFModelFiles(("model.gguf",)),
        model_config,
    )

    assert name_map == {
        "blk.40.nextn.eh_proj.weight": "mtp.fc.weight",
        "blk.40.attn_q.weight": "mtp.layers.0.self_attn.q_proj.weight",
    }


@pytest.mark.parametrize(
    "name",
    [
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
    ],
)
def test_mtp_transform_undoes_norm_offset(name):
    model_config = SimpleNamespace(hf_config=PretrainedConfig())
    output = dict(
        Qwen35MtpGGUFAdapter().transform_weights(
            [(name, torch.full((8,), 3.0))],
            model_config,
        )
    )
    assert torch.equal(output[name], torch.full((8,), 2.0))


def test_mtp_transform_keeps_quantized_params():
    model_config = SimpleNamespace(hf_config=PretrainedConfig())
    weights = [
        ("mtp.layers.0.self_attn.q_proj.qweight", torch.ones(8, 4)),
        ("mtp.layers.0.self_attn.q_proj.qweight_type", torch.tensor(14)),
    ]
    output = dict(Qwen35MtpGGUFAdapter().transform_weights(weights, model_config))
    assert torch.equal(
        output["mtp.layers.0.self_attn.q_proj.qweight"],
        torch.ones(8, 4),
    )
    assert output["mtp.layers.0.self_attn.q_proj.qweight_type"] == 14
