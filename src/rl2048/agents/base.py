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
    BackwardRollout,
    EpochReport,
    ForwardRollout,
    Report,
    Rollout,
    TrainCarry,
)
from rl2048.jumanji import Game2048, Observation, State, TimeStep, Viewer
from rl2048.policies import Policy
from rl2048.utils import tree_select


class Agent(eqx.Module):
    key: PRNGKey
    policy: Policy
    optim: GradientTransformation
    opt_state: OptState
    batch_size: int | None
    env: Game2048
    state: State

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        key: PRNGKey,
        policy: Policy,
        optim: GradientTransformation | None = None,
        batch_size: int | None = None,
        board_size: int = 4,
        viewer: Viewer | None = None,
    ) -> None:
        new_key, reset_key = jax.random.split(key)
        self.key = new_key

        self.policy = policy

        self.optim = optim
        self.opt_state = (
            None
            if self.optim is None
            else self.optim.init(eqx.filter(self.params_dict(), eqx.is_array))
        )

        self.batch_size = batch_size
        self.env = Game2048(board_size, viewer)
        self.state, _ = self._reset(reset_key)

    @property
    def params(self) -> Sequence["str"]:
        raise NotImplementedError

    def train_epoch(
        self, epoch: int | None = None, **kwargs: Any
    ) -> tuple[Self, EpochReport]:
        raise NotImplementedError

    def params_dict(self) -> dict[str, PyTree]:
        return {param: getattr(self, param) for param in self.params}

    def set_optim(self, optim: GradientTransformation) -> Self:
        return self.replace(
            optim=optim,
            optim_state=optim.init(eqx.filter(self.params_dict(), eqx.is_array)),
        )

    def update_params(self, grads: Self) -> Self:
        params = self.params_dict()
        param_grads = self.__class__.params_dict(grads)
        updates, opt_state = self.optim.update(param_grads, self.opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        return self.replace(opt_state=opt_state, **new_params)

    def replace(self, **kwargs: Any) -> Self:
        def where(agent: Self) -> tuple:
            return tuple(getattr(agent, key) for key in kwargs)

        replace = tuple(val for val in kwargs.values())
        new_agent: Self = eqx.tree_at(where, self, replace)
        return new_agent

    def reset(self, key: PRNGKey | None = None) -> Self:
        if key is None:
            key = self.key

        new_key, reset_key = jax.random.split(key)
        state, _ = self._reset(reset_key)

        return self.replace(key=new_key, state=state)

    def _reset(self, key: PRNGKey) -> tuple[State, TimeStep]:
        return (
            self.env.reset(key)
            if self.batch_size is None
            else jax.vmap(self.env.reset)(jax.random.split(key, self.batch_size))
        )

    def step(self, action: chex.Array | None = None, *, inference: bool = True) -> Self:
        next_key, sample_key = jax.random.split(self.key)

        if action is None:
            action = self.sample(
                sample_key, self.state.observation(), inference=inference
            )

        state, _ = (
            self.env.step(self.state, action)
            if self.batch_size is None
            else jax.vmap(self.env.step)(self.state, action)
        )

        return self.replace(key=next_key, state=state)

    def sample(
        self, key: PRNGKey, observation: Observation, *, inference: bool = True
    ) -> Array:
        if self.batch_size is None:
            sample = partial(self.policy.sample, inference=inference)
        else:
            key = jax.random.split(key, self.batch_size)
            sample = partial(self.policy.__class__.sample, inference=inference)
            sample = partial(eqx.filter_vmap(sample, in_axes=(None, 0, 0)), self.policy)
        return sample(key, observation)

        return self._sample(key, observation, inference=inference)

    def render(self) -> Any:
        return self.env.render(self.state)

    def train(self, n_epochs: int, **kwargs: Any) -> Report:
        epoch_index = jnp.arange(n_epochs)
        init_params, static = eqx.partition(AgentReport.init(self), eqx.is_array)
        carry = TrainCarry(init_params, init_params)
        train_epoch = partial(self.__class__._train_epoch, static, **kwargs)  # noqa: SLF001
        train_epoch = partial(eqx.filter_jit(train_epoch))

        carry, epochs = jax.lax.scan(train_epoch, carry, epoch_index)

        return Report(
            last=eqx.combine(carry.curr, static),
            best=eqx.combine(carry.best, static),
            epochs=epochs,
        )

    @staticmethod
    def _train_epoch(
        static: AgentReport, carry: TrainCarry, epoch: int, **kwargs: Any
    ) -> tuple[TrainCarry, EpochReport]:
        curr = eqx.combine(carry.curr, static)
        agent, report = curr.agent.train_epoch(epoch=epoch, **kwargs)
        is_best = report.avg_score > carry.best.epoch.avg_score
        next_curr = AgentReport(agent=agent, epoch=report)
        next_curr_params, _ = eqx.partition(next_curr, eqx.is_array)
        next_best_params = tree_select(is_best, next_curr_params, carry.best)
        next_carry = TrainCarry(curr=next_curr_params, best=next_best_params)
        return next_carry, report

    def rollout(self, n_steps: int, *, inference: bool = True) -> tuple[Self, Rollout]:
        new_key, forward_key = jax.random.split(self.key)

        # Play game to generate forward rollout
        forward_rollout = self.__class__._forward_rollout  # noqa: SLF001
        forward_rollout = partial(forward_rollout, inference=inference)
        if self.batch_size is None:
            forward_rollout = partial(eqx.filter_jit(forward_rollout), self)
            forward_keys = jax.random.split(forward_key, n_steps)
        else:
            forward_rollout = partial(
                eqx.filter_vmap(forward_rollout, in_axes=(None, 0, 0)), self
            )
            forward_keys = jax.random.split(
                forward_key, self.batch_size * n_steps
            ).reshape(n_steps, self.batch_size, 2)
        forward: ForwardRollout
        final_state, forward = jax.lax.scan(forward_rollout, self.state, forward_keys)

        # Loop back over the forward rollout to populate final_state
        backward_rollout = self._backward_rollout
        if self.batch_size is not None:
            backward_rollout = eqx.filter_vmap(self._backward_rollout)
        final = forward.at(-1).backward()
        reverse_forward = forward.at(slice(None, None, -1))
        _, reverse_backward = jax.lax.scan(backward_rollout, final, reverse_forward)
        backward = reverse_backward.at(slice(None, None, -1))

        return self.replace(key=new_key, state=final_state), backward

    def _forward_rollout(
        self, state: State, key: PRNGKey, *, inference: bool
    ) -> tuple[State, ForwardRollout]:
        sample = partial(self.policy.sample, inference=inference)
        action = sample(key, state.observation())
        next_state, timestep = self.env.step(state, action)
        if timestep.extras is None:
            msg = "Timestep must have extra dict containing next_state."
            raise RuntimeError(msg)
        step = ForwardRollout(
            timestep.step_type,
            state,
            action,
            jnp.asarray(timestep.reward),
            timestep.extras["next_state"],
        )
        return next_state, step

    @classmethod
    def _backward_rollout(
        cls, final: BackwardRollout, curr: ForwardRollout
    ) -> tuple[BackwardRollout, Rollout]:
        next_final = tree_select(curr.last(), curr.backward(), final)
        return next_final, Rollout(*curr, *next_final)
