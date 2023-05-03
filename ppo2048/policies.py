from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
from jumanji.env import Environment

from ppo2048.types import EnvState, NetworkParams, Observation, Step

if TYPE_CHECKING:
    from ppo2048.networks import Networks


class Policy(ABC):
    @abstractmethod
    def neglogprobs(
        self,
        observation: Observation,
        *args: tuple[chex.ArrayTree],
        **kwargs: dict["str", chex.ArrayTree],
    ) -> chex.Array:
        raise NotImplementedError

    @abstractmethod
    def value(
        self,
        observation: Observation,
        *args: tuple[chex.ArrayTree],
        **kwargs: dict["str", chex.ArrayTree],
    ) -> chex.Array:
        raise NotImplementedError

    def apply(
        self,
        observation: Observation,
        *args: tuple[chex.ArrayTree],
        **kwargs: dict["str", chex.ArrayTree],
    ) -> chex.Array:
        neglogprobs = self.neglogprobs(observation, *args, **kwargs)
        value = self.value(observation, *args, **kwargs)
        return neglogprobs, value

    def evaluate(
        self,
        observation: Observation,
        action: chex.Array,
        *args: tuple[chex.ArrayTree],
        **kwargs: dict["str", chex.ArrayTree],
    ) -> tuple[chex.Array, chex.Array]:
        neglogprobs, value = self.apply(observation, *args, **kwargs)
        neglogprob = neglogprobs[action]
        return neglogprob, value

    def choose(
        self,
        key: chex.PRNGKey,
        observation: Observation,
        *args: tuple[chex.ArrayTree],
        **kwargs: dict["str", chex.ArrayTree],
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        del key
        neglogprobs, value = self.apply(observation, *args, **kwargs)
        action = jnp.argmin(neglogprobs, -1)
        neglogprob = neglogprobs[action]
        return action, neglogprob, value

    def rollout(
        self,
        env: Environment,
        n_steps: int,
        key: chex.PRNGKey,
        init_state: EnvState,
        *args: tuple[chex.ArrayTree],
        **kwargs: tuple[chex.ArrayTree],
    ) -> tuple[EnvState, Step]:
        def step(state: EnvState, key: chex.PRNGKey) -> tuple[EnvState, Step]:
            observation = Observation(state.board, state.action_mask)
            action, neglogprob, value = self.choose(key, observation, *args, **kwargs)
            state, timestep = env.step(state, action)
            transition = Step(
                observation,
                action,
                neglogprob,
                value,
                timestep.reward,
                timestep.discount,
            )
            return state, transition

        keys = jax.random.split(key, n_steps)
        return jax.lax.scan(step, init_state, keys)


class RandomPolicy(Policy):
    def neglogprobs(self, observation: Observation) -> chex.Array:
        neglogprob = -jnp.log(1.0 / observation.action_mask.sum(-1))
        masked_neglogprob = jnp.where(observation.action_mask, neglogprob, jnp.inf)
        return masked_neglogprob

    def value(self, observation: Observation) -> chex.Array:
        batch_size = observation.board.shape[0]
        return jnp.zeros(batch_size)

    def choose(
        self, key: chex.PRNGKey, observation: Observation
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        batch_size = observation.board.shape[0]
        keys = jax.random.split(key, batch_size)
        actions = jnp.tile(jnp.arange(4, dtype=int), (batch_size, 1))
        action = jax.vmap(jax.random.choice)(keys, actions, p=observation.action_mask)
        neglogprob, value = self.evaluate(observation, action)
        return action, neglogprob, value


class NaivePolicy(Policy):
    def neglogprobs(self, observation: Observation) -> chex.Array:
        action = jnp.argmax(observation.action_mask * jnp.array((1, 3, 4, 2)))
        neglogprobs = jnp.full_like(observation.action_mask, jnp.inf)
        batch_size = observation.board.shape[0]
        neglogprobs = neglogprobs.at[jnp.arange(batch_size, dtype=int), action].set(0)
        return neglogprobs

    def value(self, observation: Observation) -> chex.Array:
        batch_size = observation.board.shape[0]
        return jnp.zeros(batch_size)


class NetworkPolicy(Policy):
    def __init__(self, networks: "Networks") -> None:
        self.networks = networks

    def neglogprobs(
        self, observation: Observation, params: NetworkParams
    ) -> chex.Array:
        return self.networks.apply_policy(params, observation)

    def value(self, observation: Observation, params: NetworkParams) -> chex.Array:
        return self.networks.apply_value(params, observation)

    def apply(self, observation: Observation, params: NetworkParams) -> chex.Array:
        return self.networks.apply(params, observation)
