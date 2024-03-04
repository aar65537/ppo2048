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


import equinox as eqx
import jax
from chex import PRNGKey
from jaxtyping import Array

from rl2048.critics.base import Critic
from rl2048.embedders import Embedder


class DeepCritic(Critic):
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
                eqx.nn.Linear(32, 1, key=linear_key_2),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
            ]
        )

    def __call__(self, board: Array, key: PRNGKey | None = None) -> Array:
        x = self.embedder(board)
        x = self.dropout(x, key=key)
        return self.network(x)[0]
