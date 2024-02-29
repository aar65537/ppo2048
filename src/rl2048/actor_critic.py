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

from collections.abc import Callable

import equinox as eqx

from rl2048.critics import Critic
from rl2048.policies import Policy


class ActorCritic(eqx.Module):
    actor_critic: eqx.nn.Shared

    def __init__(
        self, actor: Policy, critic: Critic, where: Callable, get: Callable
    ) -> None:
        self.actor_critic = eqx.nn.Shared((actor, critic), where, get)

    def actor(self) -> Policy:
        actor: Policy = self.actor_critic()[0]
        return actor

    def critic(self) -> Critic:
        critic: Critic = self.actor_critic()[1]
        return critic

    def policy(self) -> Policy:
        return self.actor()
