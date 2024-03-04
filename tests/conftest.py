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
from optax import GradientTransformation, adam
from rl2048.game import Game, Observation


@pytest.fixture()
def key() -> PRNGKey:
    return jax.random.PRNGKey(0)


@pytest.fixture()
def env(key: PRNGKey) -> Game:
    return Game(key)


@pytest.fixture()
def board() -> Array:
    return jnp.array([[1, 1, 2, 2], [3, 4, 0, 0], [0, 2, 0, 0], [0, 5, 0, 0]])


@pytest.fixture()
def observation(env: Game, board: Array) -> Observation:
    action_mask = jnp.asarray(env._env._get_action_mask(board))
    return Observation(board, action_mask)


@pytest.fixture()
def n_epochs() -> int:
    return 10


@pytest.fixture()
def optim() -> GradientTransformation:
    return adam(0.01)
