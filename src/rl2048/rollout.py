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
from functools import partial

import equinox as eqx
import jax

from rl2048.game import Game
from rl2048.policies.base import Policy
from rl2048.types import TimeStep


def rollout(game: Game, policy: Policy, n_steps: int) -> tuple[Game, Policy, TimeStep]:
    carry, static = eqx.partition((game, policy), eqx.is_array)
    step = partial(_step, static)
    (next_game, next_policy), timesteps = jax.lax.scan(step, carry, None, n_steps)
    return next_game, next_policy, timesteps


@eqx.filter_jit
def _step(
    static: tuple[Game, Policy], carry: tuple[Game, Policy], _: None
) -> tuple[tuple[Game, Policy], TimeStep]:
    game, policy = eqx.combine(static, carry)
    next_policy, action, neglogprob = policy.sample(game.observation)
    next_game, reward, next_obs = game.step(action)
    timestep = TimeStep(
        obs=game.observation,
        action=action,
        neglogprob=neglogprob,
        reward=reward,
        next_obs=next_obs,
    )
    carry = eqx.filter((next_game, next_policy), eqx.is_array)
    return carry, timestep
