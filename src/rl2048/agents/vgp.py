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

from typing import Self, override

import equinox as eqx
import jax
import jax.numpy as jnp
from chex import PRNGKey
from jaxtyping import Array, PyTree

from rl2048.agents.actor_critic import ActorCriticAgent
from rl2048.agents.types import EpochReport, Rollout


class VGPAgent(ActorCriticAgent):
    @override
    def train_epoch(self, epoch: int = 0) -> tuple[Self, EpochReport]:
        self, rollout = self.rollout()
        next_key, loss_key = jax.random.split(self.key)

        loss = eqx.filter_vmap(self.__class__.loss, in_axes=(None, 0, 0))
        if self.batch_size is None:
            loss_key = jax.random.split(loss_key, self.rollout_size)
        else:
            loss = eqx.filter_vmap(loss, in_axes=(None, 0, 0))
            loss_key = jax.random.split(loss_key, self.batch_size * self.rollout_size)
            loss_key = loss_key.reshape(self.rollout_size, self.batch_size, 2)

        def _mean_loss(
            _agent: Self, _rollout: Rollout, _loss_key: PRNGKey
        ) -> tuple[Array, dict[str, PyTree]]:
            loss_value, loss_extras = loss(_agent, _rollout, _loss_key)
            return jnp.mean(loss_value), jax.tree_map(jnp.mean, loss_extras)

        mean_loss = eqx.filter_value_and_grad(_mean_loss, has_aux=True)
        grads: Self
        (loss_value, loss_extras), grads = mean_loss(self, rollout, loss_key)

        self = self.update_params(grads).replace(key=next_key)

        batch_size = 1 if self.batch_size is None else self.batch_size
        report = EpochReport(
            epoch=jnp.asarray(epoch, int),
            n_steps=jnp.asarray(self.rollout_size * batch_size, int),
            n_games=rollout.n_games(),
            avg_score=rollout.avg_score(),
            high_score=rollout.high_score(),
            max_tile=rollout.max_tile(),
            extras=dict(loss=loss_value, **loss_extras),
        )

        return self, report

    def loss(
        self, step: Rollout, key: PRNGKey | None = None
    ) -> tuple[Array, dict[str, PyTree]]:
        if key is None:
            actor_key, critic_key = None, None
        else:
            actor_key, critic_key = jax.random.split(key)
        del key

        observation = step.state.observation()
        log_prob = self.policy.log_prob(observation, step.action, actor_key)
        del actor_key
        value = self.critic(observation.board, critic_key)
        del critic_key

        advantage: Array = step.extras["advantage"]
        discount_return: Array = step.extras["discount_return"]
        actor_loss = -(log_prob * advantage)
        critic_loss = -((value - discount_return) ** 2)

        extras = {"actor_loss": actor_loss, "critic_loss": critic_loss}
        return actor_loss + critic_loss, extras

    @staticmethod
    def _epoch_extras() -> dict[str, PyTree]:
        return {
            "loss": jnp.zeros((), float),
            "actor_loss": jnp.zeros((), float),
            "critic_loss": jnp.zeros((), float),
        }
