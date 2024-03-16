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

from typing import Any, NamedTuple, Self

import jax
import jax.numpy as jnp
from jaxtyping import Array


class Observation(NamedTuple):
    board: Array  # (board_size, board_size)[int32]
    action_mask: Array  # (4,)[bool]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.board.shape[:-2]

    @property
    def board_size(self) -> int:
        return self.board.shape[-1]

    def max_tile(self) -> Array:
        return 2 ** self.board.max((-2, -1))

    def terminal(self) -> Array:
        return jnp.logical_not(self.action_mask.any(-1))

    @classmethod
    def generate(cls, board_size: int) -> Self:
        board = jnp.zeros((board_size, board_size), jnp.int32)
        action_mask = jnp.zeros(4, jnp.bool)
        return cls(board, action_mask)


class TimeStep(NamedTuple):
    obs: Observation
    action: Array
    neglogprob: Array
    reward: Array
    next_obs: Observation

    def last(self) -> Array:
        return self.next_obs.terminal()

    def at(self, key: Any, /) -> "TimeStep":
        rollout: TimeStep = jax.tree_map(lambda x: x[key], self)
        return rollout
