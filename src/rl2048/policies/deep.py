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

from rl2048.jumanji import Observation
from rl2048.policies.base import Policy


class DeepPolicy(Policy):
    dropout: eqx.nn.Dropout
    embedder: eqx.nn.Sequential
    mlp: eqx.nn.Sequential
    n_tiles: int
    n_features: int

    def __init__(self, key: PRNGKey, board_size: int = 4) -> None:
        self.dropout = eqx.nn.Dropout()
        self.n_tiles = board_size**2 + 2
        key, conv_key_1, conv_key_2, conv_key_3 = jax.random.split(key, 4)
        self.embedder = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(self.n_tiles, 32, (2, 2), key=conv_key_1),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
                eqx.nn.Conv2d(32, 64, (2, 2), key=conv_key_2),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
                eqx.nn.Conv2d(64, 128, (2, 2), key=conv_key_3),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
            ]
        )
        self.n_features = self._embed(jnp.zeros((board_size, board_size))).shape[0]
        key, linear_key_1, linear_key_2 = jax.random.split(key, 3)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(self.n_features, 32, key=linear_key_1),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
                eqx.nn.Linear(32, 4, key=linear_key_2),
            ]
        )

    @override
    def __call__(
        self,
        observation: Observation,
        key: PRNGKey | None = None,
        *,
        inference: bool = True,
    ) -> Array:
        x = self._embed(observation.board)
        x = self.dropout(x.flatten(), key=key, inference=inference)
        x = self.mlp(x)
        return jax.nn.softmax(jnp.where(observation.action_mask, x, -jnp.inf))

    def _embed(self, board: Array) -> Array:
        x = jax.nn.one_hot(board, self.n_tiles, axis=0)
        return self.embedder(x)
