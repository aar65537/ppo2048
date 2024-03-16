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

import equinox as eqx
import jax
import jax.numpy as jnp
import oopax
from chex import PRNGKey
from jaxtyping import Array

from rl2048.types import Observation


class Policy(eqx.Module):
    """Abstract policy class."""

    key: eqx.AbstractVar[PRNGKey]

    @abstractmethod
    def _call(self, key: PRNGKey, obs: Observation) -> tuple[oopax.MapTree, Array]:
        raise NotImplementedError

    @eqx.filter_jit
    @oopax.capture_update
    @oopax.consume_key
    def __call__(self, key: PRNGKey, obs: Observation) -> tuple[oopax.MapTree, Array]:
        key = jax.random.split(key, obs.batch_shape)
        call = oopax.auto_vmap(self._call, lambda key: key.shape[:-1])
        return call(key, obs)

    @eqx.filter_jit
    @oopax.capture_update
    @oopax.consume_key
    def sample(
        self, key: PRNGKey, obs: Observation
    ) -> tuple[oopax.MapTree, Array, Array]:
        key = jax.random.split(key, obs.batch_shape)
        sample = oopax.auto_vmap(self._sample, lambda key: key.shape[:-1])
        return sample(key, obs)

    def _sample(
        self, key: PRNGKey, obs: Observation
    ) -> tuple[oopax.MapTree, Array, Array]:
        call_key, choice_key = jax.random.split(key)
        update, probs = self._call(call_key, obs)
        action = jax.random.choice(choice_key, jnp.arange(4, dtype=jnp.int32), p=probs)
        neglogprob = -jnp.log(probs[action])
        return update, action, neglogprob
