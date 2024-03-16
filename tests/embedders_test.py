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

import chex
import jax.numpy as jnp
import pytest
from chex import PRNGKey
from rl2048.embedders import DeepEmbedder, Embedder
from rl2048.game import Game


class EmbedderType(Enum):
    DEEP = DeepEmbedder

    def create(self, key: PRNGKey, board_size: int) -> Embedder:
        match self:
            case EmbedderType.DEEP:
                return DeepEmbedder(key, board_size)
            case _:
                msg = f"Policy type {self!r} not recognized."
                raise ValueError(msg)


pytestmark = [pytest.mark.parametrize("embedder_type", list(EmbedderType))]


def test__call__(
    key: PRNGKey, game: Game, embedder_type: EmbedderType, jit: bool
) -> None:
    with chex.fake_jit(not jit):
        embedder = embedder_type.create(key, game.board_size)
        new_embedder, embedding = embedder(game.observation)
        assert jnp.logical_not(jnp.equal(embedder.key, new_embedder.key)).all()
        assert embedding.shape == (*game.batch_shape, embedder.n_features)
