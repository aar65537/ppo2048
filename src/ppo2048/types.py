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

"""Type definitions."""

from typing import NamedTuple

import chex
import haiku as hk
from jumanji.environments.logic.game_2048.types import Observation
from jumanji.environments.logic.game_2048.types import State as EnvState


class Embedding(NamedTuple):
    """Contianer for feature embedding."""

    features: chex.Array
    action_mask: chex.Array


class NetworkParams(NamedTuple):
    """Container for network parameters."""

    embedding: hk.Params | None
    policy: hk.Params
    value: hk.Params


class AgentState(NamedTuple):
    """Container for agent state."""

    network_params: NetworkParams


class TrainingState(NamedTuple):
    """Container for training state."""

    epoch: chex.Array
    env_state: EnvState
    agent_state: AgentState


class Step(NamedTuple):
    """Container for a rollout step."""

    observation: Observation
    action: chex.Array
    neglogprob: chex.Array
    value: chex.Array
    reward: chex.Array
    discount: chex.Array


class Sample(NamedTuple):
    """Container for a training sample."""

    observation: Observation
    action: chex.Array
    neglogprob: chex.Array
    value: chex.Array
    advantage: chex.Array


class Metrics(NamedTuple):
    """Container for loss values."""

    loss: chex.Array
    policy: chex.Array
    value: chex.Array
    approxkl: chex.Array
    clipfrac: chex.Array
