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
from chex import ArrayTree, PRNGKey
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
    def count(self, coin: Array) -> dict[str, ArrayTree]:
        """Count a coin."""
        return {"n_coins": self.n_coins + 1, "total": self.total + coin}

    @eqx.filter_jit
    @mutates("object")
    def new_object(self) -> dict[str, Any]:
        """Get new object."""
        del self
        return {"object": object()}

    @eqx.filter_jit
    @mutates(key=True)
    def next_key(self, key: PRNGKey) -> None:
        """Increment key."""
        del self, key

    @eqx.filter_jit
    @mutates("total,n_coins", out=True)
    def count_and_get(self, coin: Array) -> tuple[dict[str, ArrayTree], Array]:
        """Count coin and return total."""
        new_total = self.total + coin
        return {"n_coins": self.n_coins + 1, "total": new_total}, new_total

    @eqx.filter_jit
    @mutates("n_coins,total", key=True, out=True)
    def count_and_rand(
        self, key: PRNGKey, coin: Array
    ) -> tuple[dict[str, ArrayTree], Array]:
        """Count coint and return random int."""
        rand = jax.random.randint(key, (), 1, 10**6)
        return {"n_coins": self.n_coins + 1, "total": self.total + coin}, rand


@pytest.fixture()
def counter() -> CoinCounter:
    return CoinCounter()


def test_mutates(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        for coin in range(1, 10):
            counter = counter.count(jnp.asarray(coin))
            assert counter.n_coins == coin
            assert counter.total == (coin * (coin + 1)) // 2

        eqx.clear_caches()  # type: ignore[no-untyped-call]
        jax.clear_caches()  # type: ignore[no-untyped-call]
        chex.clear_trace_counter()
        count = eqx.filter_jit(chex.assert_max_traces(counter.__class__.count, 1))

        for coin in range(10, 20):
            counter = count(counter, jnp.asarray(coin))
            assert counter.n_coins == coin
            assert counter.total == (coin * (coin + 1)) // 2


def test_mutates__object(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        prev_object = counter.object
        counter = counter.new_object()
        assert counter.object != prev_object

        eqx.clear_caches()  # type: ignore[no-untyped-call]
        jax.clear_caches()  # type: ignore[no-untyped-call]
        chex.clear_trace_counter()
        new_object = counter.__class__.new_object
        new_object = eqx.filter_jit(chex.assert_max_traces(new_object, 1))

        counter = new_object(counter)
        if jit:
            with pytest.raises(AssertionError):
                new_object(counter)
        else:
            new_object(counter)


def test_mutates__with_key(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        last_key = counter.key
        for _ in range(1, 10):
            counter = counter.next_key()
            not jnp.equal(last_key, counter.key).all()
            last_key = counter.key

        eqx.clear_caches()  # type: ignore[no-untyped-call]
        jax.clear_caches()  # type: ignore[no-untyped-call]
        chex.clear_trace_counter()
        next_key = eqx.filter_jit(chex.assert_max_traces(counter.__class__.next_key, 1))

        for _ in range(10, 20):
            counter = next_key(counter)
            not jnp.equal(last_key, counter.key).all()
            last_key = counter.key


def test_mutates__with_output(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        for coin in range(1, 10):
            counter, total = counter.count_and_get(jnp.asarray(coin))
            assert counter.n_coins == coin
            assert counter.total == (coin * (coin + 1)) // 2
            assert total == (coin * (coin + 1)) // 2

        eqx.clear_caches()  # type: ignore[no-untyped-call]
        jax.clear_caches()  # type: ignore[no-untyped-call]
        chex.clear_trace_counter()
        count_and_get = chex.assert_max_traces(counter.__class__.count_and_get, 1)
        count_and_get = eqx.filter_jit(count_and_get)

        for coin in range(10, 20):
            counter, total = count_and_get(counter, jnp.asarray(coin))
            assert counter.n_coins == coin
            assert counter.total == (coin * (coin + 1)) // 2
            assert total == (coin * (coin + 1)) // 2


def test_mutates__with_key_and_output(counter: CoinCounter, jit: bool) -> None:
    with chex.fake_jit(not jit):
        last_random = None
        for coin in range(1, 10):
            counter, random = counter.count_and_rand(jnp.asarray(coin))
            assert random != last_random
            last_random = random

        eqx.clear_caches()  # type: ignore[no-untyped-call]
        jax.clear_caches()  # type: ignore[no-untyped-call]
        chex.clear_trace_counter()
        count_random = chex.assert_max_traces(counter.__class__.count_and_rand, 1)
        count_random = eqx.filter_jit(count_random)

        for coin in range(10, 20):
            counter, random = count_random(counter, jnp.asarray(coin))
            assert random != last_random
