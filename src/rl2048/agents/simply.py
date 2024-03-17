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

import jax

from rl2048.agents.base import Agent
from rl2048.game import Game
from rl2048.policies.base import Policy
from rl2048.policies.naive import NaivePolicy


class SimpleAgent(Agent):
    game: Game
    policy: Policy

    def __init__(self, game: Game, policy: Policy) -> None:
        self.game = game
        self.policy = policy


def main() -> None:
    key = jax.random.PRNGKey(0)
    game = Game(key)
    policy = NaivePolicy()
    agent = SimpleAgent(game, policy)
    next_agent, timesteps = agent.rollout(10)
    print(timesteps)


if __name__ == "__main__":
    main()
