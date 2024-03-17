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

import chex
import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array
from rl2048.game import Game
from rl2048.policies import NaivePolicy, Policy
from rl2048.types import Observation


def test__call__(game: Game, policy: Policy, jit: bool) -> None:
    with chex.fake_jit(not jit):
        next_policy, probs = policy(game.observation)

        chex.assert_trees_all_equal_shapes_and_dtypes(
            eqx.filter(policy, eqx.is_array), eqx.filter(next_policy, eqx.is_array)
        )
        assert jnp.logical_not(jnp.equal(policy.key, next_policy.key)).all()
        chex.assert_trees_all_close(probs.sum(-1), 1)


@pytest.mark.parametrize("n_actions", [0, 1, 2, 3])
def test__call__masking(
    board: Array, policy: Policy, jit: bool, n_actions: int
) -> None:
    with chex.fake_jit(not jit):
        action_mask = jnp.zeros(4, jnp.int32).at[n_actions:].set(1)
        observation = Observation(board, action_mask)
        next_policy, probs = policy(observation)

        assert jnp.logical_not(jnp.equal(policy.key, next_policy.key)).all()
        chex.assert_trees_all_close(probs.sum(), 1)
        chex.assert_trees_all_close(probs[:n_actions], 0)
        if isinstance(policy, NaivePolicy):
            chex.assert_trees_all_close(probs[n_actions], 1)
            chex.assert_trees_all_close(probs[n_actions + 1 :], 0)
        else:
            assert all(probs[n_actions:] > 0)


def test_sample(game: Game, policy: Policy, jit: bool) -> None:
    with chex.fake_jit(not jit):
        next_policy, action, neglogprob = policy.sample(game.observation)

        chex.assert_trees_all_equal_shapes_and_dtypes(
            eqx.filter(policy, eqx.is_array), eqx.filter(next_policy, eqx.is_array)
        )
        assert jnp.logical_not(jnp.equal(policy.key, next_policy.key)).all()
        assert (action >= 0).all()
        assert (action < 4).all()
        assert (neglogprob >= 0).all()
