# Copyright 2024 the rl2048 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

import equinox as eqx
import jax
import pytest
from chex import PRNGKey
from rl2048.embedders import DeepEmbedder, Embedder
from rl2048.jumanji import Board


class EmbedderType(Enum):
    DEEP = DeepEmbedder

    def create(self, key: PRNGKey) -> Embedder:
        match self:
            case EmbedderType.DEEP:
                return DeepEmbedder(key)
            case _:
                msg = f"Policy type {self!r} not recognized."
                raise ValueError(msg)


pytestmark = [
    pytest.mark.parametrize("jit", [True, False]),
    pytest.mark.parametrize("embedder_type", list(EmbedderType)),
]


def test__call__(
    key: PRNGKey, board: Board, embedder_type: EmbedderType, jit: bool
) -> None:
    init_key, call_key = jax.random.split(key)
    del key

    embedder = embedder_type.create(init_key)
    del init_key

    call = embedder_type.value.__call__
    call = eqx.filter_jit(call) if jit else call
    embedding = call(embedder, board, call_key)
    del call_key

    assert embedding.shape == (embedder.n_features,)
