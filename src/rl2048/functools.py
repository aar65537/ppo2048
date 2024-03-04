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
from jaxtyping import Array

Leaf = Array | eqx.Module
Node = Leaf | Iterable["Node"] | Mapping[str, "Node"]
FlatTree: TypeAlias = Iterable[Node]
MapTree: TypeAlias = Mapping[str, Node]

T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")
ModuleVar = TypeVar("ModuleVar", bound=eqx.Module)
NodeVar = TypeVar("NodeVar", bound=Node)


def auto_vmap(
    fn: Callable[Concatenate[ModuleVar, P], T],
    *args: Any,
    batch_shape: str = "batch_shape",
    **kwargs: Any,
) -> Callable[Concatenate[ModuleVar, P], T]:
    @wraps(fn)
    def inner(module: ModuleVar, *in_args: P.args, **in_kwargs: P.kwargs) -> T:
        vmap_fn = fn
        for _ in getattr(module, batch_shape):
            vmap_fn = eqx.filter_vmap(vmap_fn, *args, **kwargs)
        return vmap_fn(module, *in_args, **in_kwargs)  # type: ignore[no-any-return]

    return inner


def strip_return(fn: Callable[P, tuple[T, *Ts]]) -> Callable[P, T]:
    @wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        return fn(*args, **kwargs)[0]

    return inner


def capture_attrs(
    fn: Callable[Concatenate[ModuleVar, P], tuple[MapTree, *Ts]],
) -> Callable[Concatenate[ModuleVar, P], tuple[ModuleVar, *Ts]]:
    @wraps(fn)
    def inner(
        module: ModuleVar, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[ModuleVar, *Ts]:
        update, *output = fn(module, *args, **kwargs)

        def where_fn(_module: ModuleVar) -> FlatTree:
            return [getattr(_module, attr) for attr in update]

        dynamic_update = eqx.filter(list(update.values()), eqx.is_array)
        dynamic_module, static_module = eqx.partition(module, eqx.is_array)
        dynamic_new_module = eqx.tree_at(where_fn, dynamic_module, dynamic_update)
        new_module = eqx.combine(dynamic_new_module, static_module)
        return (new_module, *output)  # type: ignore[return-value]

    return inner


def consume_attr(
    fn: Callable[Concatenate[ModuleVar, NodeVar, P], tuple[MapTree, *Ts]],
    *,
    attr: str,
    split_fn: Callable[[ModuleVar], tuple[NodeVar, NodeVar]],
) -> Callable[Concatenate[ModuleVar, P], tuple[MapTree, *Ts]]:
    @wraps(fn)
    def inner(
        module: ModuleVar, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[MapTree, *Ts]:
        next_attr, sub_attr = split_fn(module)
        update, *output = fn(module, sub_attr, *args, **kwargs)
        if attr in update:
            msg = (
                f"Update from '{fn.__qualname__}' already contains "
                f"attribute '{attr}'."
            )
            raise ValueError(msg)
        update = {attr: next_attr, **update}
        return (update, *output)  # type: ignore[return-value]

    del inner.__wrapped__  # type: ignore[attr-defined]
    return inner


def consume_key(
    fn: Callable[Concatenate[ModuleVar, PRNGKey, P], tuple[MapTree, *Ts]],
    *,
    key: str = "key",
) -> Callable[Concatenate[ModuleVar, P], tuple[MapTree, *Ts]]:
    split_key = partial(_split_key, key=key)
    return consume_attr(fn, attr=key, split_fn=split_key)


def _split_key(module: eqx.Module, key: str) -> tuple[PRNGKey, PRNGKey]:
    old_key: PRNGKey = getattr(module, key)
    del module

    split = jax.random.split
    for _ in old_key.shape[:-1]:
        split = jax.vmap(jax.random.split)

    next_keys = split(old_key)
    del old_key

    next_key = next_keys.take(0, -2)
    sub_key = next_keys.take(1, -2)
    del next_keys

    return (next_key, sub_key)
