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

"""Namespace containing modified jumanji attributes."""

from functools import cache
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array
from jumanji.environments.logic.game_2048.env import Game2048 as _Game2048
from jumanji.environments.logic.game_2048.types import State

from rl2048.functools import (
    MapTree,
    auto_vmap,
    capture_attrs,
    strip_return,
)


class Observation(NamedTuple):
    board: Array  # (board_size, board_size)[int32]
    action_mask: Array  # (4,)[bool]


class Game(eqx.Module):
    key: PRNGKey  # (2,)[uint32]
    step_count: Array  # ()[int32]
    board: Array  # (board_size, board_size)[int32]
    action_mask: Array  # (4,)[bool]
    score: Array  # ()[float32]

    def __init__(
        self,
        key: PRNGKey,
        batch_shape: tuple[int, ...] | None = None,
        board_size: int = 4,
    ) -> None:
        if batch_shape is not None:
            key = jax.random.split(key, batch_shape)

        env = _get_jumanji_env(board_size)
        state = _reset(env, key)
        del key

        self.key = jnp.asarray(state.key)
        self.step_count = jnp.asarray(state.step_count)
        self.board = jnp.asarray(state.board)
        self.action_mask = jnp.asarray(state.action_mask)
        self.score = jnp.asarray(state.score)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.step_count.shape

    @property
    def board_size(self) -> int:
        return self.board.shape[-1]

    @property
    def observation(self) -> Observation:
        return Observation(self.board, self.action_mask)

    @property
    def _env(self) -> _Game2048:
        return _get_jumanji_env(self.board_size)

    @eqx.filter_jit
    @strip_return
    @capture_attrs
    def reset(
        self,
        key: PRNGKey | None = None,
        batch_shape: tuple[int, ...] | None = None,
    ) -> tuple[MapTree]:
        if key is not None and batch_shape is not None:
            key = jax.random.split(key, batch_shape)
        elif key is not None and batch_shape is None:
            pass
        elif key is None and batch_shape is not None:
            key = self.key
            for _ in self.key.shape[:-1]:
                key = key[0]
            key = jax.random.split(key, batch_shape)
        else:
            key = self.key

        state = _reset(self._env, key)
        del key

        update = {
            field: jnp.asarray(getattr(state, field))
            for field in state  # type: ignore[attr-defined]
        }
        return (update,)

    @eqx.filter_jit
    @strip_return
    @capture_attrs
    @auto_vmap
    def step(self, action: Array) -> tuple[MapTree]:
        state = State(
            board=self.board,
            step_count=self.step_count,
            action_mask=self.action_mask,
            score=self.score,
            key=self.key,
        )
        state, _ = self._env.step(state, action)
        update = {
            field: jnp.asarray(getattr(state, field))
            for field in state  # type: ignore[attr-defined]
        }
        return (update,)

    def max_tile(self, *args: Any, **kwargs: Any) -> Array:
        return 2 ** self.board.max(*args, **kwargs)


@cache
def _get_jumanji_env(board_size: int) -> _Game2048:
    return _Game2048(board_size=board_size)  # type: ignore[return-value]


def _reset(env: _Game2048, key: PRNGKey) -> State:
    reset = env.reset
    for _ in key.shape[:-1]:
        reset = jax.vmap(reset)
    return reset(key)[0]
