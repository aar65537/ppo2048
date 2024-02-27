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


from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from rl2048.jumanji import State, StepType

if TYPE_CHECKING:
    from rl2048.agents.base import Agent


class BackwardRollout(NamedTuple):
    step_type: StepType
    state: State


class ForwardRollout(NamedTuple):
    step_type: StepType
    state: State
    action: Array
    reward: Array
    next_state: State

    def backward(self) -> BackwardRollout:
        return BackwardRollout(self.step_type, self.next_state)

    def mid(self) -> Array:
        return jnp.equal(self.step_type, StepType.MID)

    def last(self) -> Array:
        return jnp.equal(self.step_type, StepType.LAST)

    def at(self, key: Any, /) -> "ForwardRollout":
        rollout: ForwardRollout = jax.tree_map(lambda x: x[key], self)
        return rollout


class Rollout(NamedTuple):
    step_type: StepType
    state: State
    action: Array
    reward: Array
    next_state: State
    final_step_type: StepType
    final_state: State

    def mid(self) -> Array:
        return jnp.equal(self.step_type, StepType.MID)

    def last(self) -> Array:
        return jnp.equal(self.step_type, StepType.LAST)

    def finished(self) -> Array:
        return jnp.equal(self.final_step_type, StepType.LAST)

    def truncated(self) -> Array:
        return jnp.equal(self.final_step_type, StepType.MID)

    def reward_to_go(self) -> Array:
        return (self.final_state.score - self.next_state.score).astype(int)

    def total_score(self, *args: Any, **kwargs: Any) -> Array:
        return (self.final_state.score * self.last()).sum(*args, **kwargs).astype(int)

    def n_finished(self, *args: Any, **kwargs: Any) -> Array:
        return self.last().sum(*args, **kwargs)

    def avg_score(self, *args: Any, **kwargs: Any) -> Array:
        return jnp.nan_to_num(
            self.total_score(*args, **kwargs) / self.n_finished(*args, **kwargs)
        )

    def high_score(self, *args: Any, **kwargs: Any) -> Array:
        return self.final_state.score.max(*args, **kwargs).astype(int)

    def max_tile(self, *args: Any, **kwargs: Any) -> Array:
        return self.final_state.max_tile(*args, **kwargs)

    def at(self, key: Any, /) -> "Rollout":
        rollout: Rollout = jax.tree_map(lambda x: x[key], self)
        return rollout


class EpochReport(NamedTuple):
    epoch: Array
    n_steps: Array
    n_games: Array
    avg_score: Array
    high_score: Array
    max_tile: Array
    loss: Array

    def at(self, key: Any, /) -> "EpochReport":
        report: EpochReport = jax.tree_map(lambda x: x[key], self)
        return report

    @staticmethod
    def init() -> "EpochReport":
        return EpochReport(
            epoch=jnp.asarray(0, int),
            n_steps=jnp.asarray(0, int),
            n_games=jnp.asarray(0, int),
            avg_score=jnp.asarray(0, float),
            high_score=jnp.asarray(0, int),
            max_tile=jnp.asarray(0, int),
            loss=jnp.asarray(0, float),
        )


class AgentReport(NamedTuple):
    agent: "Agent"
    epoch: EpochReport

    @staticmethod
    def init(agent: "Agent") -> "AgentReport":
        return AgentReport(agent=agent, epoch=EpochReport.init())


class TrainCarry(NamedTuple):
    curr: AgentReport
    best: AgentReport


class Report(NamedTuple):
    last: AgentReport
    best: AgentReport
    epochs: EpochReport
