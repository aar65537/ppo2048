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

from collections.abc import Callable, Iterable, Mapping
from functools import partial, wraps
from typing import Any, Concatenate, ParamSpec, TypeAlias, TypeVar, TypeVarTuple

import equinox as eqx
import jax
from chex import PRNGKey
from jaxtyping import PyTree

FlatTree: TypeAlias = Iterable[PyTree]
MapTree: TypeAlias = Mapping[str, PyTree]

T = TypeVar("T")
U = TypeVar("U")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")


def auto_vmap(
    shape_fn: Callable[..., tuple[int, ...]], **vmap_kwargs: Any
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            vmap_fn = fn
            for _ in shape_fn(*args, **kwargs):
                vmap_fn = eqx.filter_vmap(vmap_fn, **vmap_kwargs)
            return vmap_fn(*args, **kwargs)  # type: ignore[no-any-return]

        return inner

    return decorator


def auto_vmap_key(
    shape_fn: Callable[..., tuple[int, ...]], *, key: str = "key", **vmap_kwargs: Any
) -> Callable[
    [Callable[Concatenate[T, PRNGKey, P], tuple[MapTree, *Ts]]],
    Callable[Concatenate[T, P], tuple[MapTree, *Ts]],
]:
    def decorator(
        fn: Callable[Concatenate[T, PRNGKey, P], tuple[MapTree, *Ts]],
    ) -> Callable[Concatenate[T, P], tuple[MapTree, *Ts]]:
        auto_vmap_dec = auto_vmap(shape_fn, **vmap_kwargs)
        split_key = partial(_split_key_auto_vmap, key=key, shape_fn=shape_fn)
        return consume_attr(auto_vmap_dec(fn), update_attr=key, split_fn=split_key)

    return decorator


def capture_update(
    fn: Callable[Concatenate[T, P], tuple[MapTree, *Ts]],
) -> Callable[Concatenate[T, P], tuple[T, *Ts]]:
    @wraps(fn)
    def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> tuple[T, *Ts]:
        update, *output = fn(module, *args, **kwargs)

        def where_fn(_module: T) -> FlatTree:
            return [getattr(_module, attr) for attr in update]

        dynamic_update = eqx.filter(list(update.values()), eqx.is_array)
        dynamic_module, static_module = eqx.partition(module, eqx.is_array)
        dynamic_new_module = eqx.tree_at(where_fn, dynamic_module, dynamic_update)
        new_module = eqx.combine(dynamic_new_module, static_module)
        return (new_module, *output)  # type: ignore[return-value]

    return inner


def consume_attr(
    fn: Callable[Concatenate[T, U, P], tuple[MapTree, *Ts]],
    *,
    update_attr: str,
    split_fn: Callable[Concatenate[T, P], tuple[U, U]],
) -> Callable[Concatenate[T, P], tuple[MapTree, *Ts]]:
    @wraps(fn)
    def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> tuple[MapTree, *Ts]:
        next_attr, sub_attr = split_fn(module, *args, **kwargs)
        update, *output = fn(module, sub_attr, *args, **kwargs)

        if update_attr in update:
            msg = (
                f"Update from '{fn.__qualname__}' already contains "
                f"attribute '{update_attr}'."
            )
            raise ValueError(msg)

        update = {update_attr: next_attr, **update}
        return (update, *output)  # type: ignore[return-value]

    del inner.__wrapped__  # type: ignore[attr-defined]
    return inner


def consume_key(
    fn: Callable[Concatenate[T, PRNGKey, P], tuple[MapTree, *Ts]],
    *,
    key: str = "key",
) -> Callable[Concatenate[T, P], tuple[MapTree, *Ts]]:
    split_key = partial(_split_key, key=key)
    return consume_attr(fn, update_attr=key, split_fn=split_key)


def strip_output(fn: Callable[P, tuple[T, *Ts]]) -> Callable[P, T]:
    @wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        return fn(*args, **kwargs)[0]

    return inner


def _split_key(
    module: PyTree, *args: Any, key: str, **kwargs: Any
) -> tuple[PRNGKey, PRNGKey]:
    del args, kwargs

    old_key: PRNGKey = getattr(module, key)
    del key, module

    split = auto_vmap(lambda key: key.shape[:-1])(jax.random.split)
    next_keys = split(old_key)
    del old_key

    next_key = next_keys.take(0, -2)
    sub_key = next_keys.take(1, -2)
    del next_keys

    return (next_key, sub_key)


def _split_key_auto_vmap(
    module: PyTree,
    *args: Any,
    key: str,
    shape_fn: Callable[..., tuple[int, ...]],
    **kwargs: Any,
) -> tuple[PRNGKey, PRNGKey]:
    old_key = getattr(module, key)
    del key

    next_key, sub_key = jax.random.split(old_key)
    del old_key

    shape = shape_fn(module, *args, **kwargs)
    del module, args, kwargs

    return next_key, jax.random.split(sub_key, shape)
