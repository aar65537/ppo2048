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


import chex
import equinox as eqx
import jax.numpy as jnp
from rl2048.embedders import Embedder
from rl2048.game import Game


def test__call__(embedder: Embedder, game: Game, jit: bool) -> None:
    with chex.fake_jit(not jit):
        next_embedder, embedding = embedder(game.observation)

        chex.assert_trees_all_equal_shapes_and_dtypes(
            eqx.filter(embedder, eqx.is_array), eqx.filter(next_embedder, eqx.is_array)
        )
        assert jnp.logical_not(jnp.equal(embedder.key, next_embedder.key)).all()
        assert embedding.shape == (*game.batch_shape, embedder.n_features)
