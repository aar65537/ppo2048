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
from rl2048.embedders import DeepEmbedder
from rl2048.jumanji import Board, Observation
from rl2048.policies import DeepPolicy, NaivePolicy, Policy, RandomPolicy


class PolicyType(Enum):
    DEEP = DeepPolicy
    NAIVE = NaivePolicy
    RANDOM = RandomPolicy

    def create(self, key: PRNGKey) -> Policy:
        match self:
            case PolicyType.DEEP:
                embedder_key, policy_key = jax.random.split(key)
                embedder = DeepEmbedder(embedder_key)
                return DeepPolicy(policy_key, embedder)
            case PolicyType.NAIVE:
                return NaivePolicy()
            case PolicyType.RANDOM:
                return RandomPolicy()
            case _:
                msg = f"Policy type {self!r} not recognized."
                raise ValueError(msg)


pytestmark = [
    pytest.mark.parametrize("jit", [True, False]),
    pytest.mark.parametrize("policy_type", list(PolicyType)),
]


@pytest.mark.parametrize("n_actions", [0, 1, 2, 3])
def test__call__(
    key: PRNGKey, board: Board, policy_type: PolicyType, n_actions: int, jit: bool
) -> None:
    init_key, call_key = jax.random.split(key)
    del key

    policy = policy_type.create(init_key)
    del init_key

    action_mask = jnp.zeros(4, jnp.int32).at[n_actions:].set(1)
    observation = Observation(board, action_mask)

    call = policy_type.value.__call__
    call = eqx.filter_jit(call) if jit else call
    probs = call(policy, observation, call_key)
    del call_key

    chex.assert_trees_all_close(probs.sum(), 1)
    chex.assert_trees_all_close(probs[:n_actions], 0)
    if policy_type == PolicyType.NAIVE:
        chex.assert_trees_all_close(probs[n_actions], 1)
        chex.assert_trees_all_close(probs[n_actions + 1 :], 0)
    else:
        assert all(probs[n_actions:] > 0)


def test_sample(
    key: PRNGKey, observation: Observation, policy_type: PolicyType, jit: bool
) -> None:
    init_key, sample_key = jax.random.split(key)
    del key

    policy = policy_type.create(init_key)
    del init_key

    sample = policy_type.value.sample
    sample = eqx.filter_jit(sample) if jit else sample
    probs, action = sample(policy, sample_key, observation)
    del sample_key

    chex.assert_trees_all_close(probs.sum(), 1)
    assert 0 <= action < 4
    assert probs[action] > 0


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_log_prob(
    key: PRNGKey,
    observation: Observation,
    policy_type: PolicyType,
    action: int,
    jit: bool,
) -> None:
    init_key, log_prob_key = jax.random.split(key)
    del key

    policy = policy_type.create(init_key)
    del init_key

    log_prob_fn = policy_type.value.log_prob
    log_prob_fn = eqx.filter_jit(log_prob_fn) if jit else log_prob_fn
    log_prob = log_prob_fn(policy, observation, action, log_prob_key)
    del log_prob_key

    if not observation.action_mask[action]:
        assert log_prob == -jnp.inf
    elif policy_type == PolicyType.NAIVE:
        assert log_prob == 0 or log_prob == -jnp.inf  # noqa: PLR1714
    else:
        assert -jnp.inf < log_prob < 0
