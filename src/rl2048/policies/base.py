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
from chex import PRNGKey
from jaxtyping import Array

from rl2048.jumanji import Observation


class Policy(eqx.Module):
    """Abstract policy class."""

    @abstractmethod
    def __call__(
        self,
        observation: Observation,
        key: PRNGKey | None = None,
        *,
        inference: bool = True,
    ) -> Array:
        raise NotImplementedError

    def log_prob(
        self,
        observation: Observation,
        action: Array,
        key: PRNGKey | None = None,
        *,
        inference: bool = True,
    ) -> Array:
        return jnp.log(self(observation, key, inference=inference)[action])

    def sample(
        self, key: PRNGKey, observation: Observation, *, inference: bool = True
    ) -> Array:
        call_key, choice_key = jax.random.split(key)
        probs = self(observation, call_key, inference=inference)
        return jax.random.choice(choice_key, jnp.arange(4, dtype=jnp.int32), p=probs)
