# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen35_gguf_with_hf_mtp_fallback():
    repo_id = "unsloth/Qwen3.5-0.8B-GGUF"
    hf_hub_download(repo_id, filename="mmproj-BF16.gguf")
    backbone = hf_hub_download(
        repo_id,
        filename="Qwen3.5-0.8B-Q4_K_M.gguf",
    )
    llm = LLM(
        model=backbone,
        tokenizer="Qwen/Qwen3.5-0.8B",
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=1024,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
        },
    )
    try:
        outputs = llm.generate(
            ["<|im_start|>user\nWhat is GGUF?<|im_end|>\n<|im_start|>assistant\n"],
            SamplingParams(temperature=0, max_tokens=8),
        )
        assert outputs[0].outputs[0].token_ids
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
