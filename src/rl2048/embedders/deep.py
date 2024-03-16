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
import oopax
from chex import PRNGKey
from jaxtyping import Array

from rl2048.embedders.base import Embedder
from rl2048.types import Observation


class DeepEmbedder(Embedder):
    key: PRNGKey
    n_features: int
    n_tiles: int
    network: eqx.nn.Sequential

    def __init__(self, key: PRNGKey, board_size: int = 4) -> None:
        self.key, call_key, conv_key_1, conv_key_2, conv_key_3 = jax.random.split(
            key, 5
        )
        self.n_tiles = board_size**2 + 2
        self.network = eqx.nn.Sequential(
            [
                eqx.nn.Conv2d(self.n_tiles, 32, (2, 2), key=conv_key_1),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
                eqx.nn.Conv2d(32, 64, (2, 2), key=conv_key_2),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
                eqx.nn.Conv2d(64, 128, (2, 2), key=conv_key_3),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
            ]
        )
        obs = Observation.generate(board_size)
        self.n_features = self._call(call_key, obs)[1].shape[0]

    def _call(self, key: PRNGKey, obs: Observation) -> tuple[oopax.MapTree, Array]:
        return {}, self.network(
            jax.nn.one_hot(obs.board, self.n_tiles, axis=0), key=key
        ).flatten()
