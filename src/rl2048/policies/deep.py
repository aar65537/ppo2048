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

from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array

from rl2048.embedders import Embedder
from rl2048.game import Observation
from rl2048.policies.base import Policy


class DeepPolicy(Policy):
    dropout: eqx.nn.Dropout
    embedder: Embedder
    network: eqx.nn.Sequential

    def __init__(self, key: PRNGKey, embedder: Embedder) -> None:
        self.dropout = eqx.nn.Dropout()
        self.embedder = embedder
        key, linear_key_1, linear_key_2 = jax.random.split(key, 3)
        self.network = eqx.nn.Sequential(
            [
                eqx.nn.Linear(self.embedder.n_features, 32, key=linear_key_1),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
                eqx.nn.Linear(32, 4, key=linear_key_2),
            ]
        )

    @override
    def __call__(self, observation: Observation, key: PRNGKey | None = None) -> Array:
        x = self.embedder(observation.board)
        x = self.dropout(x, key=key)
        x = self.network(x)
        return jax.nn.softmax(jnp.where(observation.action_mask, x, -jnp.inf))
