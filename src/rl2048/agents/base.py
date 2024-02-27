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
from jaxtyping import PyTree
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
from rl2048.jumanji import Game2048, State, Viewer
from rl2048.policies import Policy
from rl2048.utils import tree_select


class Agent(eqx.Module):
    key: PRNGKey
    env: Game2048
    state: State
    policy: Policy
    optim: GradientTransformation
    opt_state: OptState

    def __init__(  # noqa: PLR0913
        self,
        key: PRNGKey,
        policy: Policy,
        optim: GradientTransformation | None = None,
        board_size: int = 4,
        viewer: Viewer | None = None,
    ) -> None:
        new_key, reset_key = jax.random.split(key)
        self.key = new_key
        self.env = Game2048(board_size, viewer)
        self.state, _ = self.env.reset(reset_key)
        self.policy = policy
        self.optim = optim
        self.opt_state = (
            None
            if self.optim is None
            else self.optim.init(eqx.filter(self.params_dict(), eqx.is_array))
        )

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

    def reset(self, key: PRNGKey | None = None, state: State | None = None) -> Self:
        if key is not None and state is not None:
            msg = "Reset accepts key or state, not both."
            raise ValueError(msg)

        if key is None:
            key = self.key

        new_key, reset_key = jax.random.split(key)

        if state is None:
            state, _ = self.env.reset(reset_key)

        return self.replace(key=new_key, state=state)

    def step(self, action: chex.Array | None = None, *, inference: bool = True) -> Self:
        key = self.key
        if action is None:
            key, sample_key = jax.random.split(key)
            action = self.policy.sample(
                sample_key, self.state.observation(), inference=inference
            )
        state, _ = self.env.step(self.state, action)
        return self.replace(key=key, state=state)

    def render(self) -> Any:
        return self.env.render(self.state)

    def train(self, n_epochs: int, *args: Any, **kwargs: Any) -> Report:
        epoch_index = jnp.arange(n_epochs)
        init_params, static = eqx.partition(AgentReport.init(self), eqx.is_array)
        carry = TrainCarry(init_params, init_params)
        train_epoch = partial(self.__class__._train_epoch, static, *args, **kwargs)  # noqa: SLF001
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
        forward_rollout = partial(eqx.filter_jit(forward_rollout), self)
        forward_keys = jax.random.split(forward_key, n_steps)
        forward: ForwardRollout
        final_state, forward = jax.lax.scan(forward_rollout, self.state, forward_keys)

        # Loop back over the forward rollout to populate final_state
        backward_rollout = self._backward_rollout
        final = forward.at(-1).backward()
        reverse_forward = forward.at(slice(None, None, -1))
        _, reverse_backward = jax.lax.scan(backward_rollout, final, reverse_forward)
        backward = reverse_backward.at(slice(None, None, -1))

        return self.replace(key=new_key, state=final_state), backward

    def _forward_rollout(
        self, state: State, key: PRNGKey, *, inference: bool
    ) -> tuple[State, ForwardRollout]:
        action = self.policy.sample(key, state.observation(), inference=inference)
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
