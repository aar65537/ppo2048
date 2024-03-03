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
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from chex import ArrayTree, PRNGKey
from jaxtyping import Array
from rl2048.functools import capture_attrs, consume_key, strip_return

pytestmark = [
    pytest.mark.parametrize("jit", [True, False]),
]


class Counter(eqx.Module):
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
    @strip_return
    @capture_attrs
    def count(self, coin: Array) -> tuple[dict[str, ArrayTree]]:
        """Count a coin."""
        return ({"n_coins": self.n_coins + 1, "total": self.total + coin},)

    @eqx.filter_jit
    @strip_return
    @partial(capture_attrs, validate_trees=False)
    def new_object(self) -> tuple[dict[str, Any]]:
        """Get new object."""
        del self
        return ({"object": object()},)

    @eqx.filter_jit
    @strip_return
    @capture_attrs
    def new_object_fail(self) -> tuple[dict[str, Any]]:
        """Get new object."""
        del self
        return ({"object": object()},)

    @eqx.filter_jit
    @strip_return
    @capture_attrs
    @consume_key
    def next_key(self, key: PRNGKey) -> tuple[dict]:
        """Increment key."""
        del self, key
        return ({},)

    @eqx.filter_jit
    @capture_attrs
    def count_and_get(self, coin: Array) -> tuple[dict[str, Array], Array]:
        """Count coin and return total."""
        new_total = self.total + coin
        return {"n_coins": self.n_coins + 1, "total": new_total}, new_total

    @eqx.filter_jit
    @capture_attrs
    @consume_key
    def count_and_rand(
        self, key: PRNGKey, coin: Array
    ) -> tuple[dict[str, ArrayTree], Array]:
        """Count coint and return random int."""
        rand = jax.random.randint(key, (), 1, 10**6)
        return {"n_coins": self.n_coins + 1, "total": self.total + coin}, rand


@pytest.fixture()
def counter() -> Counter:
    return Counter()


def test_count(counter: Counter, jit: bool) -> None:
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
        if jit:
            with pytest.raises(AssertionError):
                count(Counter(), jnp.asarray(coin))


def test_new_object(counter: Counter, jit: bool) -> None:
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

        with pytest.raises(TypeError):
            counter.new_object_fail()


def test_next_key(counter: Counter, jit: bool) -> None:
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
        if jit:
            with pytest.raises(AssertionError):
                next_key(Counter())


def test_count_and_get(counter: Counter, jit: bool) -> None:
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
        if jit:
            with pytest.raises(AssertionError):
                count_and_get(Counter(), jnp.asarray(coin))


def test_cound_and_rand(counter: Counter, jit: bool) -> None:
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
        if jit:
            with pytest.raises(AssertionError):
                count_random(Counter(), jnp.asarray(coin))
