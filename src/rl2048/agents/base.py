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

from abc import abstractmethod
from collections.abc import Sequence
from functools import partial
from typing import Any, Self

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array, PyTree
from optax import GradientTransformation, OptState

from rl2048.agents.types import (
    AgentReport,
    BackwardCarry,
    BackwardStep,
    EpochReport,
    ForwardCarry,
    ForwardStep,
    Report,
    Rollout,
    TrainCarry,
)
from rl2048.jumanji import Game2048, State, TimeStep
from rl2048.policies import Policy
from rl2048.utils import tree_select


class Agent(eqx.Module):
    key: PRNGKey
    batch_size: int | None
    env: Game2048
    n_epochs: int
    optim: GradientTransformation | None
    opt_state: OptState | None
    rollout_size: int
    state: State

    def __init__(  # noqa: PLR0913
        self,
        key: PRNGKey,
        *,
        batch_size: int | None,
        env: Game2048,
        optim: GradientTransformation | None,
        rollout_size: int,
        n_epochs: int,
    ) -> None:
        next_key, reset_key = jax.random.split(key)
        del key
        self.key = next_key
        del next_key

        self.batch_size = batch_size
        self.env = env
        self.n_epochs = n_epochs
        self.optim = optim
        self.rollout_size = rollout_size

        self.state, _ = self._reset(reset_key)
        del reset_key

        if self.optim is None:
            self.opt_state = None
        else:
            params_dict = eqx.filter(self.params_dict(), eqx.is_array)
            self.opt_state = self.optim.init(params_dict)

    @property
    @abstractmethod
    def policy(self) -> Policy:
        raise NotImplementedError

    @abstractmethod
    def params(self) -> Sequence["str"]:
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self, epoch: int = 0) -> tuple[Self, EpochReport]:
        raise NotImplementedError

    def reset(self, key: PRNGKey | None = None) -> Self:
        if key is None:
            key = self.key

        next_key, reset_key = jax.random.split(key)
        state, _ = self._reset(reset_key)
        del reset_key

        return self.replace(key=next_key, state=state)

    def probs(self) -> tuple[Self, Array]:
        next_key, policy_key = jax.random.split(self.key)
        probs = self.policy(self.state.observation(), policy_key)
        return self.replace(key=next_key), probs

    def sample(self) -> tuple[Self, Array, Array]:
        next_key, sample_key = jax.random.split(self.key)
        sample = self.policy.__class__.sample

        if self.batch_size is not None:
            sample_key = jax.random.split(sample_key, self.batch_size)
            sample = eqx.filter_vmap(sample, in_axes=(None, 0, 0))

        probs, action = sample(self.policy, sample_key, self.state.observation())
        return self.replace(key=next_key), probs, action

    def step(self, action: chex.Array | None = None) -> Self:
        step = self.env.step if self.batch_size is None else jax.vmap(self.env.step)

        if action is None:
            self, _, action = self.sample()

        state, _ = step(self.state, action)

        return self.replace(state=state)

    def rollout(self) -> tuple[Self, Rollout]:
        next_key, forward_key, back_init_key, back_key = jax.random.split(self.key, 4)

        # Play game to generate forward rollout
        forward_rollout = self.__class__._forward_rollout  # noqa: SLF001
        if self.batch_size is None:
            forward_key = jax.random.split(forward_key, self.rollout_size)
        else:
            forward_rollout = eqx.filter_vmap(forward_rollout, in_axes=(None, 0, 0))
            forward_key = jax.random.split(
                forward_key, self.batch_size * self.rollout_size
            )
            forward_key = forward_key.reshape(self.rollout_size, self.batch_size, 2)
        forward_rollout = partial(eqx.filter_jit(forward_rollout), self)

        forward: ForwardStep
        final_state, forward = jax.lax.scan(forward_rollout, self.state, forward_key)
        del forward_key

        # Loop back over the forward rollout to populate final_state
        back_extras = self.__class__._init_back_extras  # noqa: SLF001
        back_rollout = partial(eqx.filter_jit(self.__class__._back_rollout), self)  # noqa: SLF001
        if self.batch_size is not None:
            back_extras = eqx.filter_vmap(back_extras, in_axes=(None, 0, 0))
            back_rollout = eqx.filter_vmap(back_rollout)  # type: ignore[assignment]
            back_init_key = jax.random.split(back_init_key, self.batch_size)
            back_key = jax.random.split(back_key, self.batch_size)

        final_step = forward.at(-1)
        back_carry_extras = back_extras(self, final_step, back_init_key)
        del back_init_key

        back_carry = BackwardCarry(final_step.last(), back_carry_extras, back_key)
        del back_key

        rev = slice(None, None, -1)
        backward = jax.lax.scan(back_rollout, back_carry, forward.at(rev))[1].at(rev)

        return self.replace(key=next_key, state=final_state), backward

    def train(self) -> Report:
        epoch_index = jnp.arange(self.n_epochs)
        init_agent = AgentReport.init(self, **self._epoch_extras())
        init_params, static = eqx.partition(init_agent, eqx.is_array)
        carry = TrainCarry(init_params, init_params)
        train_epoch = partial(self._train_epoch, static)
        train_epoch = partial(eqx.filter_jit(train_epoch))

        carry, epochs = jax.lax.scan(train_epoch, carry, epoch_index)

        return Report(
            last=eqx.combine(carry.curr, static),
            best=eqx.combine(carry.best, static),
            epochs=epochs,
        )

    def render(self) -> Any:
        return self.env.render(self.state)

    def params_dict(self) -> dict[str, PyTree]:
        return {param: getattr(self, param) for param in self.params()}

    def replace(self, **kwargs: Any) -> Self:
        new_agent: Self = eqx.tree_at(
            where=lambda agent: [getattr(agent, key) for key in kwargs],
            pytree=self,
            replace=kwargs.values(),
        )
        return new_agent

    def update_params(self, grads: Self) -> Self:
        if self.optim is None:
            msg = "Agent has no optimizer to update params."
            raise RuntimeError(msg)
        updates, opt_state = self.optim.update(
            grads.params_dict(),
            state=self.opt_state,
            params=self.params_dict(),
        )
        new_params = eqx.apply_updates(self.params_dict(), updates)
        return self.replace(opt_state=opt_state, **new_params)

    def _init_back_extras(self, step: ForwardStep, key: PRNGKey) -> dict[str, PyTree]:
        del self, step, key
        return {}

    def _update_back_extras(
        self, carry: BackwardCarry, step: ForwardStep, key: PRNGKey
    ) -> dict[str, PyTree]:
        del self, carry, step, key
        return {}

    @staticmethod
    def _epoch_extras() -> dict[str, PyTree]:
        return {}

    @staticmethod
    def _train_epoch(
        static: AgentReport, carry: TrainCarry, epoch: int, **kwargs: Any
    ) -> tuple[TrainCarry, EpochReport]:
        curr = AgentReport(*eqx.combine(carry.curr, static))
        agent, report = curr.agent.train_epoch(epoch, **kwargs)

        next_curr = AgentReport(agent, report)
        next_curr_params, _ = eqx.partition(next_curr, eqx.is_array)

        is_best = report.avg_score > carry.best.epoch.avg_score
        next_best_params = tree_select(is_best, next_curr_params, carry.best)

        next_carry = TrainCarry(next_curr_params, next_best_params)
        return next_carry, report

    def _reset(self, key: PRNGKey) -> tuple[State, TimeStep]:
        reset = self.env.reset
        if self.batch_size is not None:
            key = jax.random.split(key, self.batch_size)
            reset = jax.vmap(reset)
        return reset(key)

    def _forward_rollout(
        self, state: ForwardCarry, key: PRNGKey
    ) -> tuple[ForwardCarry, ForwardStep]:
        probs, action = self.policy.sample(key, state.observation())
        next_state, timestep = self.env.step(state, action)

        if timestep.extras is None:
            msg = "Timestep must have extra dict containing next_state."
            raise RuntimeError(msg)

        step = ForwardStep(
            state=state,
            probs=probs,
            action=action,
            reward=jnp.asarray(timestep.reward),
            next_state=timestep.extras["next_state"],
            step_type=timestep.step_type,
        )
        return next_state, step

    def _back_rollout(
        self, carry: BackwardCarry, step: ForwardStep
    ) -> tuple[BackwardCarry, BackwardStep]:
        _new_key, new_key, _update_key, update_key = jax.random.split(carry.key, 4)

        new_extras = self._init_back_extras(step, _new_key)
        del _new_key
        new_carry = BackwardCarry(jnp.asarray(1, bool), new_extras, new_key)
        del new_key

        update_extras = self._update_back_extras(carry, step, _update_key)
        del _update_key
        update_carry = carry.replace(key=update_key, **update_extras)
        del update_key

        next_carry = tree_select(step.last(), new_carry, update_carry)
        return next_carry, BackwardStep.combine(step, next_carry)
