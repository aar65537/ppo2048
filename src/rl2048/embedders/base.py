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

from abc import abstractmethod

import equinox as eqx
from chex import PRNGKey
from jaxtyping import Array


class Embedder(eqx.Module):
    n_features: int

    @abstractmethod
    def __call__(self, board: Array, key: PRNGKey | None = None) -> Array:
        raise NotImplementedError
