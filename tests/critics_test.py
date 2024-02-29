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
from rl2048.critics import Critic, DeepCritic
from rl2048.embedders import DeepEmbedder
from rl2048.jumanji import Board


class CriticType(Enum):
    DEEP = DeepCritic

    def create(self, key: PRNGKey) -> Critic:
        match self:
            case CriticType.DEEP:
                embedder_key, critic_key = jax.random.split(key)
                embedder = DeepEmbedder(embedder_key)
                return DeepCritic(critic_key, embedder)
        msg = f"Policy type {self!r} not recognized."
        raise ValueError(msg)


pytestmark = [
    pytest.mark.parametrize("jit", [True, False]),
    pytest.mark.parametrize("critic_type", list(CriticType)),
]


def test__call__(
    key: PRNGKey, board: Board, critic_type: CriticType, jit: bool
) -> None:
    init_key, call_key = jax.random.split(key)
    del key

    critic = critic_type.create(init_key)
    del init_key

    call = critic_type.value.__call__
    call = eqx.filter_jit(call) if jit else call
    value = call(critic, board, call_key)
    del call_key

    assert value.shape == ()
