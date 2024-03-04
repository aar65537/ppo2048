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
from typing import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array, PyTree
from optax import GradientTransformation
from rlax import td_lambda

from rl2048.actor_critic import ActorCritic
from rl2048.agents.base import Agent
from rl2048.agents.types import BackwardCarry, EpochReport, ForwardStep, Rollout
from rl2048.critics import Critic
from rl2048.game import DEFAULT_ENV, Game, State
from rl2048.policies import Policy


class ActorCriticAgent(Agent):
    actor_critic: ActorCritic
    gamma: Array
    lamb: Array

    def __init__(  # noqa: PLR0913
        self,
        key: PRNGKey,
        actor_critic: ActorCritic,
        *,
        batch_size: int | None = None,
        env: Game = DEFAULT_ENV,
        gamma: float = 0.95,
        lamb: float = 0.99,
        n_epochs: int = 16,
        optim: GradientTransformation | None = None,
        rollout_size: int = 256,
    ) -> None:
        self.actor_critic = actor_critic
        self.gamma = jnp.asarray(gamma, float)
        self.lamb = jnp.asarray(lamb, float)

        super().__init__(
            key,
            batch_size=batch_size,
            env=env,
            n_epochs=n_epochs,
            optim=optim,
            rollout_size=rollout_size,
        )

    @override
    @property
    def policy(self) -> Policy:
        return self.actor_critic.policy()

    @property
    def critic(self) -> Critic:
        return self.actor_critic.critic()

    @override
    def params(self) -> Sequence[str]:
        return ("actor_critic",)

    @override
    def train_epoch(self, epoch: int = 0) -> tuple[Self, EpochReport]:
        del epoch
        msg = f"{self.__class__} does not support training."
        raise NotImplementedError(msg)

    def advantage(self, rollout: Rollout) -> tuple[Self, Array]:
        next_key, value_key = jax.random.split(self.key)
        trajectory: State = jax.tree_util.tree_map(
            lambda obs_tm1, obs_t: jnp.concatenate([obs_tm1, obs_t[None]], axis=0),
            rollout.state,
            rollout.at(-1).next_state,
        )

        value_fn = jax.vmap(eqx.filter_jit(self.critic))
        if self.batch_size is None:
            value_keys = jax.random.split(value_key, self.rollout_size + 1)
        else:
            shape = (self.rollout_size + 1, self.batch_size)
            value_keys = jax.random.split(value_key, shape).reshape(*shape, 2)
            value_fn = jax.vmap(value_fn)
        del value_key

        values = value_fn(trajectory.board, value_keys)
        del value_keys
        discounts = jnp.asarray(self.gamma * rollout.mid(), float)
        value_tm1 = values[:-1]
        value_t = values[1:]

        advantage_fn = partial(td_lambda, lambda_=self.lamb, stop_target_gradients=True)
        if self.batch_size is not None:
            advantage_fn = jax.vmap(advantage_fn, in_axes=1, out_axes=1)
        advantage = advantage_fn(value_tm1, rollout.reward, discounts, value_t)

        return self.replace(key=next_key), advantage

    @override
    def _init_back_extras(self, step: ForwardStep, key: PRNGKey) -> dict[str, PyTree]:
        value_key, next_value_key = jax.random.split(key)

        value = self.critic(step.state.board, value_key)
        next_value = self.critic(step.next_state.board, next_value_key) * step.mid()
        delta = step.reward + self.gamma * next_value - value
        advantage = delta
        discount_return = step.reward + self.gamma * next_value

        return {"advantage": advantage, "discount_return": discount_return}

    @override
    def _update_back_extras(
        self, carry: BackwardCarry, step: ForwardStep, key: PRNGKey
    ) -> dict[str, PyTree]:
        value_key, next_value_key = jax.random.split(key)

        value = self.critic(step.state.board, value_key)
        next_value = self.critic(step.next_state.board, next_value_key) * step.mid()
        delta = step.reward + self.gamma * next_value - value
        advantage = delta + self.gamma * self.lamb * carry.extras["advantage"]
        discount_return = step.reward + self.gamma * carry.extras["discount_return"]

        return {"advantage": advantage, "discount_return": discount_return}
