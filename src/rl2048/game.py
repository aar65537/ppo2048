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

"""2048 game module."""

from dataclasses import asdict
from functools import cache
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array
from jumanji.environments.logic.game_2048.env import Game2048 as _Game2048

from rl2048.functools import MapTree, auto_vmap, capture_update, strip_output


class Observation(NamedTuple):
    board: Array  # (board_size, board_size)[int32]
    action_mask: Array  # (4,)[bool]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.board.shape[:-2]

    @property
    def board_size(self) -> int:
        return self.board.shape[-1]

    def max_tile(self, *args: Any, **kwargs: Any) -> Array:
        return 2 ** self.board.max(*args, **kwargs)

    def terminal(self, *args: Any, **kwargs: Any) -> Array:
        return jnp.logical_not(self.action_mask.any(*args, **kwargs))


class Game(eqx.Module):
    key: PRNGKey  # (2,)[uint32]
    step_count: Array  # ()[int32]
    board: Array  # (board_size, board_size)[int32]
    action_mask: Array  # (4,)[bool]
    score: Array  # ()[float32]

    def __init__(
        self,
        key: PRNGKey,
        *,
        batch_shape: tuple[int, ...] | None = None,
        board_size: int = 4,
    ) -> None:
        if batch_shape is not None:
            key = jax.random.split(key, batch_shape)

        env = _get_jumanji_env(board_size)
        reset = auto_vmap(lambda key: key.shape[:-1])(env.reset)
        state, _ = reset(key)
        del key

        self.key = jnp.asarray(state.key)
        self.step_count = jnp.asarray(state.step_count)
        self.board = jnp.asarray(state.board)
        self.action_mask = jnp.asarray(state.action_mask)
        self.score = jnp.asarray(state.score)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.observation.batch_shape

    @property
    def board_size(self) -> int:
        return self.observation.board_size

    @property
    def observation(self) -> Observation:
        return Observation(self.board, self.action_mask)

    @property
    def _env(self) -> _Game2048:
        return _get_jumanji_env(self.board_size)

    @eqx.filter_jit
    @strip_output
    @capture_update
    def reset(self) -> tuple[MapTree]:
        reset = auto_vmap(lambda key: key.shape[:-1])(self._env.reset)
        state, _ = reset(self.key)
        return (asdict(state),)

    @eqx.filter_jit
    @capture_update
    @auto_vmap(lambda _, action: action.shape)
    def step(self, action: Array) -> tuple[MapTree, Array]:
        """Perfom action on game."""
        step = self._env.step
        state, timestep = step(self, action)  # type: ignore[arg-type]
        return asdict(state), jnp.asarray(timestep.reward)

    def max_tile(self, *args: Any, **kwargs: Any) -> Array:
        return 2 ** self.board.max(*args, **kwargs)

    def terminal(self, *args: Any, **kwargs: Any) -> Array:
        return self.observation.terminal(*args, **kwargs)


@cache
def _get_jumanji_env(board_size: int) -> _Game2048:
    return _Game2048(board_size=board_size)
