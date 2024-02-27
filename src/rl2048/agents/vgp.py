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
from typing import Any, Self, override

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from rl2048.agents.base import Agent, Rollout
from rl2048.agents.types import EpochReport


class VGPAgent(Agent):
    @override
    @property
    def params(self) -> Sequence[str]:
        return ("policy",)

    @override
    def train_epoch(
        self, epoch: int | None = None, n_steps: int = 10**3, **kwargs: Any
    ) -> tuple[Self, EpochReport]:
        del kwargs

        agent, rollout = self.rollout(n_steps, inference=False)

        loss = eqx.filter_value_and_grad(self.__class__.loss)
        grads: Self
        loss_value, grads = loss(self, rollout)

        agent = self.update_params(grads)

        batch_size = 1 if self.batch_size is None else self.batch_size
        report = EpochReport(
            epoch=jnp.asarray(0 if epoch is None else epoch, int),
            n_steps=jnp.asarray(n_steps * batch_size, int),
            n_games=rollout.n_finished(),
            avg_score=rollout.avg_score(),
            high_score=rollout.high_score(),
            max_tile=rollout.max_tile(),
            loss=loss_value,
        )

        return agent, report

    def loss(self, step: Rollout) -> Array:
        log_prob_fn = eqx.filter_vmap(self.policy.log_prob)
        if self.batch_size is not None:
            log_prob_fn = eqx.filter_vmap(log_prob_fn)
        log_prob: Array = log_prob_fn(step.state.observation(), step.action)
        weights = step.reward_to_go() * step.finished()
        return -(log_prob * weights).mean()
