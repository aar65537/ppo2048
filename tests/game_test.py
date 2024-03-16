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
import jax.numpy as jnp
from rl2048.game import Game


def test_reset(game: Game, jit: bool) -> None:
    with chex.fake_jit(not jit):
        for _ in range(10):
            new_game = game.reset()
            assert not jnp.equal(game.key, new_game.key).all()
            assert jnp.equal(game.step_count, 0).all()
            assert jnp.equal(new_game.step_count, 0).all()
            game = new_game


def test_step(game: Game, jit: bool) -> None:
    with chex.fake_jit(not jit):
        for i in range(10):
            action = jnp.ones(game.batch_shape, int)
            new_game, reward, next_obs = game.step(action)
            assert not jnp.equal(game.key, new_game.key).all()
            assert jnp.equal(game.step_count, i).all()
            assert jnp.equal(new_game.step_count, i + 1).all()
            assert jnp.equal(reward, new_game.score - game.score).all()
            assert jnp.equal(new_game.board, next_obs.board).all()
            assert jnp.equal(new_game.action_mask, next_obs.action_mask).all()
            game = new_game
