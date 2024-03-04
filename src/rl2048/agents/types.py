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


from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array, PyTree

from rl2048.game import State, StepType

if TYPE_CHECKING:
    from rl2048.agents.base import Agent


ForwardCarry: TypeAlias = State


class BackwardCarry(NamedTuple):
    terminates: Array
    extras: dict[str, PyTree]
    key: PRNGKey

    def replace(self, **kwargs: PyTree) -> "BackwardCarry":
        key = self.key
        terminates = self.terminates
        extras = self.extras
        for _key, _value in kwargs.items():
            match _key:
                case "key":
                    key = _value
                case "terminates":
                    terminates = _value
                case _:
                    extras[_key] = _value
        return BackwardCarry(terminates, extras, key)


class ForwardStep(NamedTuple):
    state: State
    probs: Array
    action: Array
    reward: Array
    next_state: State
    step_type: StepType

    def mid(self) -> Array:
        return jnp.equal(self.step_type, StepType.MID)

    def last(self) -> Array:
        return jnp.equal(self.step_type, StepType.LAST)

    def at(self, key: Any, /) -> "ForwardStep":
        rollout: ForwardStep = jax.tree_map(lambda x: x[key], self)
        return rollout


class BackwardStep(NamedTuple):
    state: State
    probs: Array
    action: Array
    reward: Array
    next_state: State
    step_type: StepType
    terminates: Array
    extras: dict[str, PyTree]

    def mid(self) -> Array:
        return jnp.equal(self.step_type, StepType.MID)

    def last(self) -> Array:
        return jnp.equal(self.step_type, StepType.LAST)

    def truncates(self) -> Array:
        return jnp.logical_not(self.terminates)

    def n_games(self, *args: Any, **kwargs: Any) -> Array:
        return self.last().sum(*args, **kwargs)

    def avg_score(self, *args: Any, **kwargs: Any) -> Array:
        total_score = (self.next_state.score * self.last()).sum(*args, **kwargs)
        n_games = self.n_games(*args, **kwargs)
        return jnp.nan_to_num(total_score / n_games)

    def high_score(self, *args: Any, **kwargs: Any) -> Array:
        return self.next_state.score.max(*args, **kwargs)

    def max_tile(self, *args: Any, **kwargs: Any) -> Array:
        return self.next_state.max_tile(*args, **kwargs)

    def at(self, key: Any, /) -> "BackwardStep":
        rollout: BackwardStep = jax.tree_map(lambda x: x[key], self)
        return rollout

    @staticmethod
    def combine(step: ForwardStep, carry: BackwardCarry) -> "BackwardStep":
        return BackwardStep(*step, *carry[:2])


Rollout: TypeAlias = BackwardStep


class EpochReport(NamedTuple):
    epoch: Array
    n_steps: Array
    n_games: Array
    avg_score: Array
    high_score: Array
    max_tile: Array
    extras: dict[str, PyTree]

    @staticmethod
    def init(**kwags: PyTree) -> "EpochReport":
        return EpochReport(
            epoch=jnp.asarray(0, int),
            n_steps=jnp.asarray(0, int),
            n_games=jnp.asarray(0, int),
            avg_score=jnp.asarray(0, float),
            high_score=jnp.asarray(0, float),
            max_tile=jnp.asarray(0, int),
            extras=kwags,
        )

    def at(self, key: Any, /) -> "EpochReport":
        report: EpochReport = jax.tree_map(lambda x: x[key], self)
        return report


class AgentReport(NamedTuple):
    agent: "Agent"
    epoch: EpochReport

    @staticmethod
    def init(agent: "Agent", **kwargs: PyTree) -> "AgentReport":
        return AgentReport(agent=agent, epoch=EpochReport.init(**kwargs))


class TrainCarry(NamedTuple):
    curr: AgentReport
    best: AgentReport


class Report(NamedTuple):
    last: AgentReport
    best: AgentReport
    epochs: EpochReport
