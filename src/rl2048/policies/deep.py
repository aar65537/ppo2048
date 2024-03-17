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

from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import oopax
from chex import PRNGKey
from jaxtyping import Array

from rl2048.embedders import Embedder
from rl2048.policies.base import Policy
from rl2048.types import Observation


class DeepPolicy(Policy):
    key: PRNGKey
    dropout: eqx.nn.Dropout
    embedder: Embedder
    network: eqx.nn.Sequential

    def __init__(self, key: PRNGKey, embedder: Embedder) -> None:
        self.key, linear_key_1, linear_key_2 = jax.random.split(key, 3)

        self.dropout = eqx.nn.Dropout()
        self.embedder = embedder

        self.network = eqx.nn.Sequential(
            [
                eqx.nn.Linear(self.embedder.n_features, 32, key=linear_key_1),
                eqx.nn.Lambda(jax.nn.relu),  # type: ignore[call-arg]
                eqx.nn.Linear(32, 4, key=linear_key_2),
            ]
        )

    @override
    @eqx.filter_jit
    @oopax.capture_update
    @oopax.consume_key
    def __call__(self, key: PRNGKey, obs: Observation) -> tuple[oopax.MapTree, Array]:
        key = jax.random.split(key, obs.batch_shape)
        call = oopax.auto_vmap(self._call, lambda key: key.shape[:-1])
        next_embedder, embedding = self.embedder(obs)
        return {"embedder": next_embedder}, call(key, obs, embedding)

    @override
    @eqx.filter_jit
    @oopax.capture_update
    @oopax.consume_key
    def sample(
        self, key: PRNGKey, obs: Observation
    ) -> tuple[oopax.MapTree, Array, Array]:
        key = jax.random.split(key, obs.batch_shape)
        next_embedder, embedding = self.embedder(obs)
        sample = oopax.auto_vmap(self._sample, lambda key: key.shape[:-1])
        return {"embedder": next_embedder}, *sample(key, obs, embedding)

    def _call(self, key: PRNGKey, obs: Observation, embedding: Array) -> Array:
        x = self.dropout(embedding, key=key)
        x = self.network(x)
        return jax.nn.softmax(jnp.where(obs.action_mask, x, -jnp.inf))

    def _sample(
        self, key: PRNGKey, obs: Observation, embedding: Array
    ) -> tuple[Array, Array]:
        call_key, choice_key = jax.random.split(key)
        probs = self._call(call_key, obs, embedding)
        action = jax.random.choice(choice_key, jnp.arange(4, dtype=jnp.int32), p=probs)
        neglogprob = -jnp.log(probs[action])
        return action, neglogprob
