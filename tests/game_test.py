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
import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey
from rl2048.game import Game


@pytest.fixture(params=[True, False], ids=["jit", "no jit"])
def jit(request: pytest.FixtureRequest) -> bool:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[0, 1], ids=["seed=0", "seed=1"])
def key(request: pytest.FixtureRequest) -> PRNGKey:
    return jax.random.PRNGKey(request.param)


@pytest.fixture(
    params=[None, (), (10,), (5, 3)],
    ids=["shape=None", "shape=()", "shape=(10,)", "shape=(5,3)"],
)
def shape(request: pytest.FixtureRequest) -> tuple[int, ...] | None:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[4, 5, 6], ids=["board=4", "board=5", "board=6"])
def board_size(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture()
def game(key: PRNGKey, shape: tuple[int, ...], board_size: int) -> Game:
    return Game(key=key, batch_shape=shape, board_size=board_size)


def test_reset(
    game: Game,
    jit: bool,
) -> None:
    with chex.fake_jit(not jit):
        for _ in range(10):
            new_game = game.reset()
            assert not jnp.equal(game.key, new_game.key).all()
            assert jnp.equal(game.step_count, 0).all()
            assert jnp.equal(new_game.step_count, 0).all()


def test_step(game: Game, jit: bool) -> None:
    with chex.fake_jit(not jit):
        for _ in range(10):
            action = jnp.ones(game.batch_shape, int)
            new_game = game.step(action)
            assert not jnp.equal(game.key, new_game.key).all()
            assert jnp.equal(game.step_count + 1, new_game.step_count).all()
