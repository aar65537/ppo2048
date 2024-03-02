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


from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from chex import PRNGKey
from jaxtyping import Array
from rl2048.decorators import mutates

pytestmark = [
    pytest.mark.parametrize("jit", [True, False]),
]


class CoinCounter(eqx.Module):
    key: PRNGKey
    n_coins: Array
    total: Array
    object: Any

    def __init__(self) -> None:
        self.key = jax.random.PRNGKey(0)
        self.n_coins = jnp.asarray(0)
        self.total = jnp.asarray(0)
        self.object = object()

    @eqx.filter_jit
    @mutates("n_coins,total")
    def count(self, coin: Array) -> dict[str, Array]:
        return {"n_coins": self.n_coins + 1, "total": self.total + coin}

    @eqx.filter_jit
    @mutates("object")
    def new_object(self) -> dict[str, Any]:
        del self
        return {"object": object()}

    @eqx.filter_jit
    @mutates("n_coins,total", key=True, out=True)
    def count_random(self, key: PRNGKey) -> tuple[dict[str, Array], Array]:
        rand = jax.random.randint(key, (), 1, 10**6)
        return {"n_coins": self.n_coins + 1, "total": self.total + rand}, rand


@pytest.fixture()
def counter() -> CoinCounter:
    return CoinCounter()


def test_count(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        for coin in range(1, 10):
            counter = counter.count(jnp.asarray(coin))
            assert counter.n_coins == coin
            assert counter.total == (coin * (coin + 1)) // 2

        count = eqx.filter_jit(chex.assert_max_traces(counter.__class__.count, 1))
        chex.clear_trace_counter()
        for coin in range(10, 20):
            counter = count(counter, jnp.asarray(coin))
            assert counter.n_coins == coin
            assert counter.total == (coin * (coin + 1)) // 2


def test_new_object(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        prev_object = counter.object
        counter = counter.new_object()
        assert counter.object != prev_object

        new_object = counter.__class__.new_object
        new_object = eqx.filter_jit(chex.assert_max_traces(new_object, 1))
        chex.clear_trace_counter()
        counter = new_object(counter)

        if jit:
            with pytest.raises(AssertionError):
                new_object(counter)


def test_count_random(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        last_random = None
        for _ in range(1, 10):
            counter, random = counter.count_random()
            assert random != last_random
            last_random = random

        count_random = chex.assert_max_traces(counter.__class__.count_random, 1)
        count_random = eqx.filter_jit(count_random)
        chex.clear_trace_counter()
        for _ in range(10, 20):
            counter, random = count_random(counter)
            assert random != last_random
            last_random = random


CoinCounter().count_random()
