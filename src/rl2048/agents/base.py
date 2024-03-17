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

import equinox as eqx
import jax
import oopax

from rl2048.game import Game
from rl2048.policies import Policy
from rl2048.types import TimeStep


class Agent(eqx.Module):
    game: eqx.AbstractVar[Game]
    policy: eqx.AbstractVar[Policy]

    # @abstractmethod
    # def train_epoch(self, epoch: int = 0) -> tuple[Self, EpochReport]:
    #     raise NotImplementedError

    @oopax.capture_update
    def rollout(self, n_steps: int) -> tuple[oopax.MapTree, TimeStep]:
        next_agent, timesteps = jax.lax.scan(self.__class__._step, self, None, n_steps)
        return {"game": next_agent.game, "policy": next_agent.policy}, timesteps

    @oopax.capture_update
    def _step(self, _: None) -> tuple[oopax.MapTree, TimeStep]:
        next_policy, action, neglogprob = self.policy.sample(self.game.observation)
        next_game, reward, next_obs = self.game.step(action)
        timestep = TimeStep(
            obs=self.game.observation,
            action=action,
            neglogprob=neglogprob,
            reward=reward,
            next_obs=next_obs,
        )
        return ({"game": next_game, "policy": next_policy}, timestep)

    #     next_key, forward_key, back_init_key, back_key = jax.random.split(self.key, 4)

    #     # Play game to generate forward rollout
    #     forward_rollout = self.__class__._forward_rollout  # noqa: SLF001
    #     if self.batch_size is None:
    #         forward_key = jax.random.split(forward_key, self.rollout_size)
    #     else:
    #         forward_rollout = eqx.filter_vmap(forward_rollout, in_axes=(None, 0, 0))
    #         forward_key = jax.random.split(
    #             forward_key, self.batch_size * self.rollout_size
    #         )
    #         forward_key = forward_key.reshape(self.rollout_size, self.batch_size, 2)
    #     forward_rollout = partial(eqx.filter_jit(forward_rollout), self)

    #     forward: ForwardStep
    #     final_state, forward = jax.lax.scan(forward_rollout, self.state, forward_key)
    #     del forward_key

    #     # Loop back over the forward rollout to populate final_state
    #     back_extras = self.__class__._init_back_extras  # noqa: SLF001
    #     back_rollout = partial(eqx.filter_jit(self.__class__._back_rollout), self)  # noqa: SLF001
    #     if self.batch_size is not None:
    #         back_extras = eqx.filter_vmap(back_extras, in_axes=(None, 0, 0))
    #         back_rollout = eqx.filter_vmap(back_rollout)  # type: ignore[assignment]
    #         back_init_key = jax.random.split(back_init_key, self.batch_size)
    #         back_key = jax.random.split(back_key, self.batch_size)

    #     final_step = forward.at(-1)
    #     back_carry_extras = back_extras(self, final_step, back_init_key)
    #     del back_init_key

    #     back_carry = BackwardCarry(final_step.last(), back_carry_extras, back_key)
    #     del back_key

    #     rev = slice(None, None, -1)
    #     backward = jax.lax.scan(back_rollout, back_carry, forward.at(rev))[1].at(rev)

    #     return self.replace(key=next_key, state=final_state), backward

    # def train(self) -> Report:
    #     epoch_index = jnp.arange(self.n_epochs)
    #     init_agent = AgentReport.init(self, **self._epoch_extras())
    #     init_params, static = eqx.partition(init_agent, eqx.is_array)
    #     carry = TrainCarry(init_params, init_params)
    #     train_epoch = partial(self._train_epoch, static)
    #     train_epoch = partial(eqx.filter_jit(train_epoch))

    #     carry, epochs = jax.lax.scan(train_epoch, carry, epoch_index)

    #     return Report(
    #         last=eqx.combine(carry.curr, static),
    #         best=eqx.combine(carry.best, static),
    #         epochs=epochs,
    #     )

    # @staticmethod
    # def _epoch_extras() -> dict[str, PyTree]:
    #     return {}

    # @staticmethod
    # def _train_epoch(
    #     static: AgentReport, carry: TrainCarry, epoch: int, **kwargs: Any
    # ) -> tuple[TrainCarry, EpochReport]:
    #     curr = AgentReport(*eqx.combine(carry.curr, static))
    #     agent, report = curr.agent.train_epoch(epoch, **kwargs)

    #     next_curr = AgentReport(agent, report)
    #     next_curr_params, _ = eqx.partition(next_curr, eqx.is_array)

    #     is_best = report.avg_score > carry.best.epoch.avg_score
    #     next_best_params = tree_select(is_best, next_curr_params, carry.best)

    #     next_carry = TrainCarry(next_curr_params, next_best_params)
    #     return next_carry, report

    # def _forward_rollout(
    #     self, state: ForwardCarry, key: PRNGKey
    # ) -> tuple[ForwardCarry, ForwardStep]:
    #     probs, action = self.policy.sample(key, state.observation())
    #     next_state, timestep = self.env.step(state, action)

    #     if timestep.extras is None:
    #         msg = "Timestep must have extra dict containing next_state."
    #         raise RuntimeError(msg)

    #     step = ForwardStep(
    #         state=state,
    #         probs=probs,
    #         action=action,
    #         reward=jnp.asarray(timestep.reward),
    #         next_state=timestep.extras["next_state"],
    #         step_type=timestep.step_type,
    #     )
    #     return next_state, step

    # def _back_rollout(
    #     self, carry: BackwardCarry, step: ForwardStep
    # ) -> tuple[BackwardCarry, BackwardStep]:
    #     _new_key, new_key, _update_key, update_key = jax.random.split(carry.key, 4)

    #     new_extras = self._init_back_extras(step, _new_key)
    #     del _new_key
    #     new_carry = BackwardCarry(jnp.asarray(1, bool), new_extras, new_key)
    #     del new_key

    #     update_extras = self._update_back_extras(carry, step, _update_key)
    #     del _update_key
    #     update_carry = carry.replace(key=update_key, **update_extras)
    #     del update_key

    #     next_carry = tree_select(step.last(), new_carry, update_carry)
    #     return next_carry, BackwardStep.combine(step, next_carry)
