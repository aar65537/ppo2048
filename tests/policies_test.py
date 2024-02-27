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

from enum import Enum

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey
from rl2048.jumanji import Board, Observation
from rl2048.policies import DeepPolicy, NaivePolicy, Policy, RandomPolicy


class PolicyType(Enum):
    DEEP = DeepPolicy
    NAIVE = NaivePolicy
    RANDOM = RandomPolicy

    def create(self, key: PRNGKey) -> Policy:
        if self == PolicyType.DEEP:
            return DeepPolicy(key)
        if self == PolicyType.NAIVE:
            return NaivePolicy()
        if self == PolicyType.RANDOM:
            return RandomPolicy()
        msg = f"Policy type {self!r} not recognized."
        raise ValueError(msg)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("n_actions", [0, 1, 2, 3])
@pytest.mark.parametrize("policy_type", list(PolicyType))
def test_policy(
    key: PRNGKey, board: Board, policy_type: PolicyType, n_actions: int, jit: bool
) -> None:
    init_key, test_key = jax.random.split(key)
    policy = policy_type.create(init_key)
    action_mask = jnp.zeros(4, jnp.int32).at[n_actions:].set(1)
    observation = Observation(board, action_mask)
    call = (
        eqx.filter_jit(policy_type.value.__call__)
        if jit
        else policy_type.value.__call__
    )
    probs = call(policy, observation, key=test_key, inference=False)

    chex.assert_trees_all_close(probs.sum(), 1)
    chex.assert_trees_all_close(probs[:n_actions], 0)
    if policy_type == PolicyType.NAIVE:
        chex.assert_trees_all_close(probs[n_actions], 1)
        chex.assert_trees_all_close(probs[n_actions + 1 :], 0)
    else:
        assert all(probs[n_actions:] > 0)
