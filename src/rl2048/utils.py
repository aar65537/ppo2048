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
from functools import partial
from typing import TypeVar

import jax
from jaxtyping import Array

T = TypeVar("T")


def _leaf_select(cond: Array, x: T, y: T) -> T:
    selected: T = jax.lax.cond(cond, lambda _x, _: _x, lambda _, _y: _y, x, y)  # type: ignore[no-untyped-call]
    return selected


def tree_select(cond: Array, x: T, y: T) -> T:
    selected: T = jax.tree_map(partial(_leaf_select, cond), x, y)
    return selected
