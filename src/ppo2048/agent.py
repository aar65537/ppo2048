# Copyright 2023-2024 The PPO2048 Authors
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

"""Agent logic."""

import chex
import jax
from jumanji.env import Environment
from jumanji.environments.logic.game_2048.types import Observation, State

from ppo2048.networks import Networks
from ppo2048.policies import NetworkPolicy
from ppo2048.types import AgentState, Metrics, Step, TrainingState


class Agent:
    def __init__(
        self,
        env: Environment[State],
        networks: Networks,
        n_envs: int,
        n_steps: int,
        total_steps: int,
        minibatches: int,
    ) -> None:
        self.env = env
        self.networks = networks
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_minibatches = minibatches
        self.total_steps = total_steps

        self.batch_size = self.n_envs * self.n_steps
        self.minibatch_size = self.batch_size // self.n_minibatches
        self.n_epochs = self.total_steps // self.batch_size
        self.policy = NetworkPolicy(networks)

    def init(self, key: chex.PRNGKey) -> AgentState:
        env_key, network_key = jax.random.split(key, 2)
        env_keys = jax.random.split(env_key, self.n_envs)
        env_state, _ = self.env.reset(env_keys)
        network_params = self.networks.init(network_key)
        return AgentState(0, env_state, network_params)

    def rollout(self, key: chex.PRNGKey, state: TrainingState) -> tuple[State, Step]:
        return self.policy.rollout(
            self.env, self.n_steps, key, state.env_state, state.network_params
        )

    def run_epoch(
        self, key: chex.PRNGKey, state: AgentState
    ) -> tuple[AgentState, Metrics]:
        last_state, transitions = self.rollout(key, state)
        last_observation = Observation(last_state.board, last_state.action_mask)

    def learn(self, key: chex.PRNGKey, state: AgentState):
        while state.epoch < self.n_epochs:
            key, subkey = jax.random.split(key, 2)
            state, metrics = self.run_epoch(subkey, state)
