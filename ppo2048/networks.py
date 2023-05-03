from functools import partial
from typing import Any, Callable, NamedTuple, Sequence

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from ppo2048.types import Embedding, NetworkParams, Observation


def _cnn_fn(n_features: int, observation: Observation) -> Embedding:
    size = observation.board.shape[-1]
    max_tile = size**2 + 2
    board = jax.nn.one_hot(observation.board, max_tile)
    torso = hk.Sequential(
        [
            hk.Conv2D(n_features, (2, 2), padding="Valid"),
            jax.nn.relu,
            hk.Conv2D(n_features, (2, 2), padding="Valid"),
            jax.nn.relu,
            hk.Conv2D(n_features, (2, 2), padding="Valid"),
            jax.nn.relu,
            hk.Flatten(),
        ]
    )
    embedding = Embedding(torso(board), observation.action_mask)
    return embedding


def _get_embedding(
    n_features: int | None, observation: Embedding | Observation
) -> Embedding:
    if n_features is None:
        assert isinstance(observation, Embedding)
        return observation
    else:
        assert isinstance(observation, Observation)
        return _cnn_fn(n_features, observation)


class Network(NamedTuple):
    """Networks are meant to take a batch of observations: shape (B, ...)."""

    init: Callable[[chex.PRNGKey, Any], hk.Params]
    apply: Callable[[hk.Params, Any], chex.Array]

    @staticmethod
    def transform(network_fn: Callable[[Any], chex.ArrayTree]) -> "Network":
        init, apply = hk.without_apply_rng(hk.transform(network_fn))
        return Network(init, apply)

    @staticmethod
    def make_embedding(n_features: int) -> "Network":
        network_fn = partial(_cnn_fn, n_features)
        return Network.transform(network_fn)

    @staticmethod
    def make_policy(
        mlp_units: Sequence[int], n_features: int | None = None
    ) -> "Network":
        def network_fn(observation: Embedding | Observation) -> chex.Array:
            embedding = _get_embedding(n_features, observation)
            head = hk.nets.MLP((*mlp_units, 4), activate_final=False)
            neglogprobs = head(embedding.features)
            masked_neglogprobs = jnp.where(embedding.action_mask, neglogprobs, jnp.inf)
            total_neglogporb = -jax.lax.reduce(
                -masked_neglogprobs, -jnp.inf, jnp.logaddexp, (1,)
            )
            norm_neglogprob = masked_neglogprobs - total_neglogporb[..., None]
            return norm_neglogprob

        return Network.transform(network_fn)

    @staticmethod
    def make_value(
        mlp_units: Sequence[int], n_features: int | None = None
    ) -> "Network":
        def network_fn(observation: Embedding | Observation) -> chex.Array:
            embedding = _get_embedding(n_features, observation)
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            value = jnp.squeeze(head(embedding.features), -1)
            return value

        return Network.transform(network_fn)


class Networks(NamedTuple):
    """Defines the policy-value networks with optional preprocess network. The
    assumption is that the networks are given a batchof observations."""

    embedding: Network | None
    policy: Network
    value: Network

    @staticmethod
    def make(
        mlp_units: Sequence[int], n_features: int, shared_features: bool
    ) -> "Networks":
        if shared_features:
            preprocess = Network.make_embedding(n_features)
            policy = Network.make_policy(mlp_units)
            value = Network.make_value(mlp_units)
        else:
            preprocess = None
            policy = Network.make_policy(mlp_units, n_features)
            value = Network.make_value(mlp_units, n_features)
        return Networks(preprocess, policy, value)

    def init(self, key: chex.PRNGKey) -> NetworkParams:
        embedding_key, policy_key, value_key = jax.random.split(key, 3)
        observation = Observation(jnp.zeros((1, 4, 4)), jnp.zeros((1, 4)))
        if self.embedding is None:
            embedding_params = None
            policy_params = self.policy.init(policy_key, observation)
            value_params = self.value.init(value_key, observation)
        else:
            embedding_params = self.embedding.init(embedding_key, observation)
            embedding = self.embedding.apply(embedding_params, observation)
            policy_params = self.policy.init(policy_key, embedding)
            value_params = self.value.init(value_key, embedding)
        return NetworkParams(embedding_params, policy_params, value_params)

    def apply(
        self, params: NetworkParams, observation: Observation
    ) -> tuple[chex.Array, chex.Array]:
        if self.embedding is not None:
            observation = self.apply_embedding(params, observation)
        neglogprobs = self.policy.apply(params.policy, observation)
        value = self.value.apply(params.value, observation)
        return neglogprobs, value

    def apply_embedding(
        self, params: NetworkParams, observation: Observation
    ) -> Embedding:
        if self.embedding is None:
            raise ValueError("No shared embedding network.")
        embedding = self.embedding.apply(params.embedding, observation)
        return embedding

    def apply_policy(
        self, params: NetworkParams, observation: Observation
    ) -> chex.Array:
        if self.embedding is not None:
            observation = self.apply_embedding(params, observation)
        neglogprobs = self.policy.apply(params.policy, observation)
        return neglogprobs

    def apply_value(
        self, params: NetworkParams, observation: Observation
    ) -> chex.Array:
        if self.embedding is not None:
            observation = self.apply_embedding(params, observation)
        value = self.value.apply(params.value, observation)
        return value
