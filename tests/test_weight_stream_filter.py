# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np

from vllm_gguf_plugin import weight_utils


class _FakeTensor:
    def __init__(self, name: str, *, payload_allowed: bool) -> None:
        self.name = name
        self.tensor_type = SimpleNamespace(name="F32")
        self._payload_allowed = payload_allowed

    @property
    def data(self) -> np.ndarray:
        if not self._payload_allowed:
            raise AssertionError(f"payload for {self.name} was materialized")
        return np.ones((1,), dtype=np.float32)


def test_gguf_prefix_filter_runs_before_payload_materialization(monkeypatch) -> None:
    reader = SimpleNamespace(
        tensors=[
            _FakeTensor("ple.weight", payload_allowed=False),
            _FakeTensor("model.weight", payload_allowed=True),
        ]
    )
    monkeypatch.setattr(weight_utils.gguf, "GGUFReader", lambda _: reader)

    weights = list(
        weight_utils.gguf_quant_weights_iterator_multi(
            ["unused.gguf"],
            {
                "ple.weight": "model.layers.1.ple.ple_embedding.weight",
                "model.weight": "model.layers.1.mlp.weight",
            },
            exclude_prefixes=("model.layers.1.ple.ple_embedding.",),
        )
    )

    assert [name for name, _ in weights] == ["model.layers.1.mlp.weight"]
