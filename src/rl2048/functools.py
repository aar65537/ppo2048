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

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Concatenate, ParamSpec, TypeAlias, TypeVar, TypeVarTuple

import equinox as eqx
import jax
from chex import ArrayTree, PRNGKey

T = TypeVar("T", bound=eqx.Module)
U = TypeVar("U", bound=ArrayTree)
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")

MapTree: TypeAlias = Mapping[str, ArrayTree]
FlatTree: TypeAlias = list[ArrayTree]


def strip_return(
    fn: Callable[Concatenate[T, P], tuple[T, *Ts]],
) -> Callable[Concatenate[T, P], T]:
    @wraps(fn)
    def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> T:
        return fn(module, *args, **kwargs)[0]

    return inner


def capture_attrs(
    fn: Callable[Concatenate[T, P], tuple[MapTree, *Ts]], *, validate_trees: bool = True
) -> Callable[Concatenate[T, P], tuple[T, *Ts]]:
    @wraps(fn)
    def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> tuple[T, *Ts]:
        updates, *outputs = fn(module, *args, **kwargs)
        updates_flat = list(updates.values())

        def where_fn(_module: T) -> FlatTree:
            return [getattr(_module, name) for name in updates]

        if validate_trees:
            _validate_tree(updates_flat, caller=fn.__qualname__)
            _validate_tree(where_fn(module), caller=fn.__qualname__)

        new_module: T = eqx.tree_at(where_fn, module, updates_flat)
        return (new_module, *outputs)  # type: ignore[return-value]

    return inner


def consume_attr(
    fn: Callable[Concatenate[T, U, P], tuple[MapTree, *Ts]],
    *,
    attr_name: str,
    split_fn: Callable[[U], tuple[U, U]],
) -> Callable[Concatenate[T, P], tuple[MapTree, *Ts]]:
    @wraps(fn)
    def inner(module: T, *args: P.args, **kwargs: P.kwargs) -> tuple[MapTree, *Ts]:
        next_attr, sub_attr = split_fn(getattr(module, attr_name))
        updates, *outputs = fn(module, sub_attr, *args, **kwargs)
        updates = {name: value for name, value in updates.items() if name != attr_name}
        updates = {attr_name: next_attr, **updates}
        return (updates, *outputs)  # type: ignore[return-value]

    del inner.__wrapped__  # type: ignore[attr-defined]
    return inner


def consume_key(
    fn: Callable[Concatenate[T, PRNGKey, P], tuple[MapTree, *Ts]],
    *,
    attr_name: str = "key",
) -> Callable[Concatenate[T, P], tuple[MapTree, *Ts]]:
    return consume_attr(fn, attr_name=attr_name, split_fn=_split_key)


def _split_key(key: PRNGKey) -> tuple[PRNGKey, PRNGKey]:
    return jax.random.split(key)  # type: ignore[return-value]


def _validate_tree(tree: FlatTree, caller: str | None = None) -> None:
    if all(eqx.is_array(leaf) for leaf in jax.tree_flatten(tree)[0]):
        return
    msg = (
        f"Attributes captured with {capture_attrs.__qualname__} must be ArrayTrees."
        if caller is None
        else f"Attributes captured from {caller} must be ArrayTrees."
    )
    raise TypeError(msg)
