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


from typing import Any, NamedTuple, TypeAlias, override

import chex
import jax
from chex import PRNGKey, dataclass
from jaxtyping import Array
from jumanji.env import State as AnyState
from jumanji.environments.logic.game_2048.env import Game2048 as _Game2048
from jumanji.environments.logic.game_2048.types import State as _State
from jumanji.environments.logic.game_2048.viewer import Game2048Viewer as MatplotViewer
from jumanji.types import StepType
from jumanji.types import TimeStep as _TimeStep
from jumanji.viewer import Viewer as _Viewer
from jumanji.wrappers import AutoResetWrapper as _AutoResetWrapper
from jumanji.wrappers import Wrapper

Board: TypeAlias = Array


class Observation(NamedTuple):
    board: Board
    action_mask: Array


@dataclass
class State(_State):
    board: Board  # (board_size, board_size)[int32]
    step_count: Array  # ()[int32]
    action_mask: Array  # (4,)[bool]
    score: Array  # ()[float32]
    key: PRNGKey  # (2,)[uint32]

    def max_tile(self, *args: Any, **kwargs: Any) -> Array:
        return 2 ** self.board.max(*args, **kwargs)

    def observation(self) -> Observation:
        return Observation(self.board, self.action_mask)


TimeStep: TypeAlias = _TimeStep[Observation]
Viewer: TypeAlias = _Viewer[_State]


class AutoResetWrapper(_AutoResetWrapper):
    @override
    def step(self, state: AnyState, action: chex.Array) -> tuple[AnyState, _TimeStep]:
        next_state, timestep = self._env.step(state, action)

        maybe_reset_state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            lambda *x: x,
            next_state,
            timestep,
        )  # type: ignore[no-untyped-call]

        extras = {} if timestep.extras is None else timestep.extras
        extras["next_state"] = next_state
        timestep = timestep.replace(extras=extras)  # type: ignore[attr-defined]

        return maybe_reset_state, timestep


class Game2048(Wrapper[State]):
    @override
    def __init__(self, board_size: int = 4, viewer: Viewer | None = None) -> None:
        env = AutoResetWrapper(_Game2048(board_size, viewer))
        super().__init__(env)

    @override
    def reset(self, key: PRNGKey) -> tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        return State(**state), timestep

    @override
    def step(self, state: State, action: chex.Array) -> tuple[State, TimeStep]:
        next_state, timestep = self._env.step(state, action)
        if timestep.extras is None:
            msg = "Timestep must have extra dict containing next_state."
            raise RuntimeError(msg)
        extras = timestep.extras
        extras["next_state"] = State(**extras["next_state"])
        timestep = timestep.replace(extras=extras)  # type: ignore[attr-defined]
        return State(**next_state), timestep


DEFAULT_ENV = Game2048()

__all__ = [
    "DEFAULT_ENV",
    "Board",
    "Game2048",
    "MatplotViewer",
    "Observation",
    "State",
    "StepType",
    "Viewer",
]
