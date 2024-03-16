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

import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey
from jaxtyping import Array
from rl2048.game import Game, _get_jumanji_env  # noqa: PLC2701
from rl2048.types import Observation


@pytest.fixture(params=[True, False], ids=["jit", "no jit"])
def jit(request: pytest.FixtureRequest) -> bool:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture()
def board() -> Array:
    return jnp.array([[1, 1, 2, 2], [3, 4, 0, 0], [0, 2, 0, 0], [0, 5, 0, 0]])


@pytest.fixture()
def obs(board: Array) -> Observation:
    action_mask = jnp.asarray(_get_jumanji_env(board.shape[-1])._get_action_mask(board))
    return Observation(board, action_mask)


@pytest.fixture(params=[0], ids=["seed=0"])
def key(request: pytest.FixtureRequest) -> PRNGKey:
    return jax.random.PRNGKey(request.param)


@pytest.fixture(
    params=[(), (10,), (5, 3)], ids=["shape=()", "shape=(10,)", "shape=(5,3)"]
)
def shape(request: pytest.FixtureRequest) -> tuple[int, ...] | None:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[4, 5], ids=["board=4", "board=5"])
def board_size(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture()
def game(key: PRNGKey, shape: tuple[int, ...], board_size: int) -> Game:
    return Game(key=key, batch_shape=shape, board_size=board_size)
