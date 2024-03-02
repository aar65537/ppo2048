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
# ruff: noqa: E731 N801

from collections.abc import Callable
from functools import wraps
from typing import (
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    overload,
)

import equinox as eqx
import jax
from chex import PRNGKey
from jaxtyping import PyTree

T = TypeVar("T")
In = ParamSpec("In")
Out = TypeVarTuple("Out")


class T_K_Protocol(Protocol):
    key: PRNGKey


T_K = TypeVar("T_K", bound=T_K_Protocol)


PureFn_: TypeAlias = Callable[Concatenate[T, In], PyTree]
PureFn_K: TypeAlias = Callable[Concatenate[T_K, PRNGKey, In], PyTree | None]
PureFn_O: TypeAlias = Callable[Concatenate[T, In], tuple[PyTree, *Out]]
PureFn_KO: TypeAlias = Callable[
    Concatenate[T_K, PRNGKey, In], tuple[PyTree | None, *Out]
]
ImpureFn: TypeAlias = Callable[Concatenate[T, In], T]
ImpureFn_K: TypeAlias = ImpureFn[T_K, In]
ImpureFn_O: TypeAlias = Callable[Concatenate[T, In], tuple[T, *Out]]
ImpureFn_KO: TypeAlias = ImpureFn_O[T_K, In, *Out]


class MutableDecorator_(Protocol):
    def __call__(self, pure_fn: PureFn_[T, In]) -> ImpureFn[T, In]:
        ...


class MutableDecorator_K(Protocol):
    def __call__(self, pure_fn: PureFn_K[T_K, In]) -> ImpureFn_K[T_K, In]:
        ...


class MutableDecorator_O(Protocol):
    def __call__(self, pure_fn: PureFn_O[T, In, *Out]) -> ImpureFn_O[T, In, *Out]:
        ...


class MutableDecorator_KO(Protocol):
    def __call__(self, pure_fn: PureFn_KO[T_K, In, *Out]) -> ImpureFn_KO[T_K, In, *Out]:
        ...


MutableDecorator: TypeAlias = (
    MutableDecorator_ | MutableDecorator_K | MutableDecorator_O | MutableDecorator_KO
)


@overload
def mutates(
    where: str, *, key: Literal[False] = False, out: Literal[False] = False
) -> MutableDecorator_:
    ...


@overload
def mutates(
    where: str | None = None, *, key: Literal[True], out: Literal[False] = False
) -> MutableDecorator_K:
    ...


@overload
def mutates(
    where: str, *, key: Literal[False] = False, out: Literal[True]
) -> MutableDecorator_O:
    ...


@overload
def mutates(
    where: str | None = None, *, key: Literal[True], out: Literal[True]
) -> MutableDecorator_KO:
    ...


def mutates(
    where: str | None = None,
    *,
    key: Literal[True, False] = False,
    out: Literal[True, False] = False,
) -> MutableDecorator:
    attr_names = _make_attr_names(where, key=key)
    where_fn = lambda pytree: [getattr(pytree, name) for name in attr_names]

    def decorator_(pure_fn: PureFn_[T, In]) -> ImpureFn[T, In]:
        @wraps(pure_fn)
        def impure_fn(pytree: T, *args: In.args, **kwargs: In.kwargs) -> T:
            updates = pure_fn(pytree, *args, **kwargs)
            updates_flat = _flatten_updates(updates, attr_names)
            return eqx.tree_at(where_fn, pytree, updates_flat)  # type: ignore[no-any-return]

        return impure_fn

    def decorator_k(pure_fn: PureFn_K[T_K, In]) -> ImpureFn[T_K, In]:
        # @wraps(pure_fn)
        def impure_fn(pytree: T_K, *args: In.args, **kwargs: In.kwargs) -> T_K:
            next_key, sub_key = jax.random.split(pytree.key)
            updates = pure_fn(pytree, sub_key, *args, **kwargs)
            updates = {} if updates is None else updates
            updates["key"] = next_key
            updates_flat = _flatten_updates(updates, attr_names)
            return eqx.tree_at(where_fn, pytree, updates_flat)  # type: ignore[no-any-return]

        return impure_fn

    def decorator_o(pure_fn: PureFn_O[T, In, *Out]) -> ImpureFn_O[T, In, *Out]:
        @wraps(pure_fn)
        def impure_fn(pytree: T, *args: In.args, **kwargs: In.kwargs) -> tuple[T, *Out]:
            updates, *outputs = pure_fn(pytree, *args, **kwargs)
            updates_flat = _flatten_updates(updates, attr_names)
            return (eqx.tree_at(where_fn, pytree, updates_flat), *outputs)  # type: ignore[return-value]

        return impure_fn

    def decorator_ko(pure_fn: PureFn_KO[T_K, In, *Out]) -> ImpureFn_O[T_K, In, *Out]:
        # @wraps(pure_fn)
        def impure_fn(
            pytree: T_K, *args: In.args, **kwargs: In.kwargs
        ) -> tuple[T_K, *Out]:
            next_key, sub_key = jax.random.split(pytree.key)
            updates, *outputs = pure_fn(pytree, sub_key, *args, **kwargs)
            updates = {} if updates is None else updates
            updates["key"] = next_key
            updates_flat = _flatten_updates(updates, attr_names)
            return (eqx.tree_at(where_fn, pytree, updates_flat), *outputs)  # type: ignore[return-value]

        return impure_fn

    match key, out:
        case False, False:
            return decorator_
        case True, False:
            return decorator_k
        case False, True:
            return decorator_o
        case True, True:
            return decorator_ko

    msg = f"Cannot build mutates decorator with parameters {where=}, {key=}, {out=}."
    raise ValueError(msg)


def _make_attr_names(where: str | None, *, key: bool) -> list[str]:
    if where is None and not key:
        msg = f"{mutates.__name__} requires 'key' to be 'True' when 'where = None'"
        raise ValueError(msg)

    attr_names = [] if where is None else where.split(",")
    return ["key", *attr_names] if key else attr_names


def _flatten_updates(updates: dict[str, PyTree], attr_names: list[str]) -> list[PyTree]:
    return [updates[name] for name in attr_names]
