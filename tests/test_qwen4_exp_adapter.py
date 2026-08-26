# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_gguf_plugin.weights_adapter.qwen4_exp import (
    QWEN4_EXP_ARCHITECTURE,
    Qwen4ExpGGUFAdapter,
    build_qwen4_exp_name_map,
    merge_qwen4_exp_indexer_projections,
)


def test_qwen4_exp_adapter_registration_contract() -> None:
    config = SimpleNamespace(model_type="qwen4_exp")
    assert Qwen4ExpGGUFAdapter.matches(config)
    assert Qwen4ExpGGUFAdapter.architecture(config) == QWEN4_EXP_ARCHITECTURE


def test_qwen4_exp_adapter_exposes_ple_offload_prefix() -> None:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(ple_layer_ids=[2])
        )
    )

    assert Qwen4ExpGGUFAdapter().get_ple_offload_prefixes(model_config) == (
        "model.layers.1.ple.ple_embedding.",
    )


def test_qwen4_exp_name_map_covers_hc_qsa_and_ple() -> None:
    names = {
        "token_embd.weight",
        "output_hc_norm.weight",
        "blk.3.hc_attn_down.weight",
        "blk.3.indexer.q_proj.weight",
        "blk.3.indexer.k_norm.weight",
        "blk.1.ple_key.weight",
        "per_layer_token_embd.weight",
    }
    name_map, unmapped = build_qwen4_exp_name_map(
        names,
        SimpleNamespace(ple_layer_ids=[2]),
    )

    assert not unmapped
    assert name_map["token_embd.weight"] == "model.embed_tokens.weight"
    assert name_map["output_hc_norm.weight"] == (
        "model.hyper_connection_mixer.hc_norm.weight"
    )
    assert name_map["blk.3.hc_attn_down.weight"] == (
        "model.layers.3.attn_hyper_connection.input_mix_weight_down.weight"
    )
    assert name_map["blk.3.indexer.q_proj.weight"] == (
        "model.layers.3.self_attn.indexer.index_q_proj.weight"
    )
    assert name_map["blk.3.indexer.k_norm.weight"] == (
        "model.layers.3.self_attn.indexer.k_layernorm.weight"
    )
    assert name_map["blk.1.ple_key.weight"] == ("model.layers.1.ple.key_proj.weight")
    assert name_map["per_layer_token_embd.weight"] == (
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    )


def test_qwen4_exp_name_map_rejects_ambiguous_ple_layers() -> None:
    with pytest.raises(ValueError, match="exactly one PLE layer"):
        build_qwen4_exp_name_map([], SimpleNamespace(ple_layer_ids=[2, 4]))


def test_merge_qwen4_exp_indexer_quantized_projections() -> None:
    prefix = "model.layers.3.self_attn.indexer"
    quant_type = torch.tensor(23)
    weights = [
        (f"{prefix}.index_q_proj.qweight_type", quant_type),
        (f"{prefix}.index_q_proj.qweight", torch.ones((4, 3))),
        (f"{prefix}.index_k_proj.qweight_type", quant_type.clone()),
        (f"{prefix}.index_k_proj.qweight", torch.full((2, 3), 2.0)),
    ]

    merged = dict(merge_qwen4_exp_indexer_projections(weights))

    assert torch.equal(merged[f"{prefix}.index_qk_proj.qweight_type"], quant_type)
    assert torch.equal(
        merged[f"{prefix}.index_qk_proj.qweight"],
        torch.cat((torch.ones((4, 3)), torch.full((2, 3), 2.0))),
    )


def test_merge_qwen4_exp_indexer_requires_matching_quant_types() -> None:
    prefix = "model.layers.3.self_attn.indexer"
    weights = [
        (f"{prefix}.index_q_proj.qweight_type", torch.tensor(23)),
        (f"{prefix}.index_k_proj.qweight_type", torch.tensor(24)),
    ]

    with pytest.raises(ValueError, match="different quantization types"):
        list(merge_qwen4_exp_indexer_projections(weights))
