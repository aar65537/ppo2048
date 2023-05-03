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
