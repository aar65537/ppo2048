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
from optax import GradientTransformation
from rl2048.actor_critic import ActorCritic
from rl2048.agents import ActorCriticAgent, Agent, VGPAgent
from rl2048.critics import DeepCritic
from rl2048.embedders import DeepEmbedder
from rl2048.policies import DeepPolicy


class AgentType(Enum):
    ACTOR_CRITIC = ActorCriticAgent
    VGP = VGPAgent

    def create(
        self,
        key: PRNGKey,
        batch_size: int | None,
        optim: GradientTransformation | None = None,
    ) -> Agent:
        if self in {AgentType.ACTOR_CRITIC, AgentType.VGP}:
            embed_k, policy_k, critic_k, agent_k = jax.random.split(key, 4)
            embedder = DeepEmbedder(embed_k)
            policy = DeepPolicy(policy_k, embedder)
            critic = DeepCritic(critic_k, embedder)
            actor_critic = ActorCritic(policy, critic, lambda _: (), lambda _: ())
            agent: Agent = self.value(
                agent_k, actor_critic, batch_size=batch_size, optim=optim
            )
            return agent
        msg = f"Policy type {self!r} not recognized."
        raise ValueError(msg)


pytestmark = [
    pytest.mark.parametrize("jit", [True, False]),
    pytest.mark.parametrize("agent_type", list(AgentType)),
    pytest.mark.parametrize("batch_size", [None, 10]),
]


def test_reset(
    key: PRNGKey, agent_type: AgentType, batch_size: int | None, jit: bool
) -> None:
    agent = agent_type.create(key, batch_size)
    del key

    reset = agent.__class__.reset
    reset = eqx.filter_jit(reset) if jit else reset  # type: ignore[assignment]
    next_agent = reset(agent)

    assert not jnp.equal(agent.key, next_agent.key).all()
    assert not jnp.equal(agent.state.board, next_agent.state.board).all()
    assert jnp.equal(next_agent.state.step_count, 0).all()
    assert jnp.equal(next_agent.state.score, 0).all()
    assert not jnp.equal(agent.state.key, next_agent.state.key).all()


def test_sample(
    key: PRNGKey,
    agent_type: AgentType,
    batch_size: int | None,
    jit: bool,
) -> None:
    agent = agent_type.create(key, batch_size)
    del key

    sample = agent.__class__.sample
    sample = eqx.filter_jit(sample) if jit else sample
    next_agent, probs, action = sample(agent)

    assert not jnp.equal(agent.key, next_agent.key).all()
    if batch_size is None:
        chex.assert_trees_all_close(probs.sum(), 1)
        assert 0 <= action < 4
        assert probs[action] > 0
    else:
        chex.assert_trees_all_close(probs.sum(axis=1), 1)
        assert (action >= 0).all()
        assert (action < 4).all()
        for _probs, _action in zip(probs, action, strict=True):
            assert _probs[_action] > 0


def test_step(
    key: PRNGKey, agent_type: AgentType, batch_size: int | None, jit: bool
) -> None:
    agent = agent_type.create(key, batch_size)
    del key

    step = agent.__class__.step
    step = eqx.filter_jit(step) if jit else step  # type: ignore[assignment]
    next_agent = step(agent)

    assert not jnp.equal(agent.key, next_agent.key).all()
    assert not jnp.equal(agent.state.board, next_agent.state.board).all()
    assert jnp.equal(agent.state.step_count + 1, next_agent.state.step_count).all()
    assert not jnp.equal(agent.state.key, next_agent.state.key).all()


def test_rollout(
    key: PRNGKey, agent_type: AgentType, batch_size: int | None, jit: bool
) -> None:
    agent = agent_type.create(key, batch_size)
    del key

    rollout_fn = agent.__class__.rollout
    rollout_fn = eqx.filter_jit(rollout_fn) if jit else rollout_fn
    next_agent, rollout = rollout_fn(agent)

    assert not jnp.equal(agent.key, next_agent.key).all()
    assert (rollout.probs >= 0).all()
    assert (rollout.probs <= 1).all()
    assert (rollout.action >= 0).all()
    assert (rollout.action < 4).all()
    assert (rollout.reward >= 0).all()
    assert rollout.n_games() > 0
    assert rollout.avg_score() > 0
    assert rollout.high_score() > 0
    assert rollout.max_tile() > 0

    for step in range(agent.rollout_size):
        rollout_step = rollout.at(step)
        assert not jnp.equal(
            rollout_step.state.board, rollout_step.next_state.board
        ).all()


def test_train(
    key: PRNGKey,
    agent_type: AgentType,
    batch_size: int | None,
    optim: GradientTransformation,
    jit: bool,
) -> None:
    agent = agent_type.create(key, batch_size, optim)
    del key

    train = agent.__class__.train
    train = eqx.filter_jit(train) if jit else train

    try:
        report = train(agent)
    except NotImplementedError as e:
        if str(e):
            return
        msg = f"Training not implemented for {agent.__class__}"
        raise NotImplementedError(msg) from e

    params = agent.params_dict()
    next_params = report.last.agent.params_dict()
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(params, next_params)

    assert jnp.equal(report.epochs.epoch, jnp.arange(agent.n_epochs)).all()
    assert jnp.equal(
        report.epochs.n_steps,
        agent.rollout_size * (1 if batch_size is None else batch_size),
    ).all()
    assert (report.epochs.n_games > 0).all()
    assert (report.epochs.avg_score > 0).all()
    assert (report.epochs.high_score > 0).all()
    assert (report.epochs.max_tile > 0).all()


def test_advantage(
    key: PRNGKey, agent_type: AgentType, batch_size: int | None, jit: bool
) -> None:
    agent: ActorCriticAgent = agent_type.create(key, batch_size)  # type: ignore[assignment]
    del key
    agent = eqx.nn.inference_mode(agent, value=True)

    if not hasattr(agent, "advantage"):
        return

    rollout_fn = agent.__class__.rollout
    rollout_fn = eqx.filter_jit(rollout_fn) if jit else rollout_fn
    agent, rollout = rollout_fn(agent)

    agent, advantage = agent.advantage(rollout)
    # chex.assert_trees_all_close(advantage, rollout.extras["advantage"])
