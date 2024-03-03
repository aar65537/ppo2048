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

from collections.abc import Callable
from functools import wraps
from typing import (
    Any,
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
from chex import ArrayTree, PRNGKey


class TwKProtocol(Protocol):
    key: PRNGKey


T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")
TwK = TypeVar("TwK", bound=TwKProtocol)

Dict: TypeAlias = dict[str, ArrayTree]
_PureFn: TypeAlias = Callable[Concatenate[T, P], Dict]
_PureFnwK: TypeAlias = Callable[Concatenate[TwK, PRNGKey, P], Dict | None]
_PureFnwO: TypeAlias = Callable[Concatenate[T, P], tuple[Dict, *Ts]]
_PureFnwKwO: TypeAlias = Callable[Concatenate[TwK, PRNGKey, P], tuple[Dict | None, *Ts]]
_ImpureFn: TypeAlias = Callable[Concatenate[T, P], T]
_ImpureFnwO: TypeAlias = Callable[Concatenate[T, P], tuple[T, *Ts]]


class _MutatesWrapper(Protocol):
    def __call__(self, pure_fn: _PureFn[T, P]) -> _ImpureFn[T, P]:
        ...


class _MutatesWrapperwK(Protocol):
    def __call__(self, pure_fn: _PureFnwK[TwK, P]) -> _ImpureFn[TwK, P]:
        ...


class _MutatesWrapperwO(Protocol):
    def __call__(self, pure_fn: _PureFnwO[T, P, *Ts]) -> _ImpureFnwO[T, P, *Ts]:
        ...


class _MutatesWrapperwKwO(Protocol):
    def __call__(self, pure_fn: _PureFnwKwO[TwK, P, *Ts]) -> _ImpureFnwO[TwK, P, *Ts]:
        ...


MutatesWrapper: TypeAlias = (
    _MutatesWrapper | _MutatesWrapperwK | _MutatesWrapperwO | _MutatesWrapperwKwO
)


@overload
def mutates(
    where: str, *, key: Literal[False] = False, out: Literal[False] = False
) -> _MutatesWrapper:
    ...


@overload
def mutates(
    where: str | None = None, *, key: Literal[True], out: Literal[False] = False
) -> _MutatesWrapperwK:
    ...


@overload
def mutates(
    where: str, *, key: Literal[False] = False, out: Literal[True]
) -> _MutatesWrapperwO:
    ...


@overload
def mutates(
    where: str | None = None, *, key: Literal[True], out: Literal[True]
) -> _MutatesWrapperwKwO:
    ...


def mutates(
    where: str | None = None,
    *,
    key: Literal[True, False] = False,
    out: Literal[True, False] = False,
    ensure_jit: bool = True,
) -> MutatesWrapper:
    attr_names = _make_attr_names(where, key=key)

    def where_fn(pytree: Any) -> ArrayTree:
        return [getattr(pytree, name) for name in attr_names]

    def decorator_(pure_fn: _PureFn[T, P]) -> _ImpureFn[T, P]:
        @wraps(pure_fn)
        def impure_fn(pytree: T, *args: P.args, **kwargs: P.kwargs) -> T:
            updates = pure_fn(pytree, *args, **kwargs)
            _ensure_jit(where_fn, pytree, updates, ensure_jit=ensure_jit)
            updates_flat = _flatten_updates(updates, attr_names)
            new_pytree: T = eqx.tree_at(where_fn, pytree, updates_flat)
            return new_pytree

        return impure_fn

    def decorator_k(pure_fn: _PureFnwK[TwK, P]) -> _ImpureFn[TwK, P]:
        @wraps(pure_fn)
        def impure_fn(pytree: TwK, *args: P.args, **kwargs: P.kwargs) -> TwK:
            next_key, sub_key = jax.random.split(pytree.key)
            updates = pure_fn(pytree, sub_key, *args, **kwargs)
            updates = {} if updates is None else updates
            updates["key"] = next_key
            _ensure_jit(where_fn, pytree, updates, ensure_jit=ensure_jit)
            updates_flat = _flatten_updates(updates, attr_names)
            new_pytree: TwK = eqx.tree_at(where_fn, pytree, updates_flat)
            return new_pytree

        del impure_fn.__wrapped__  # type: ignore[attr-defined]
        return impure_fn

    def decorator_o(pure_fn: _PureFnwO[T, P, *Ts]) -> _ImpureFnwO[T, P, *Ts]:
        @wraps(pure_fn)
        def impure_fn(pytree: T, *args: P.args, **kwargs: P.kwargs) -> tuple[T, *Ts]:
            updates, *outputs = pure_fn(pytree, *args, **kwargs)
            _ensure_jit(where_fn, pytree, updates, ensure_jit=ensure_jit)
            updates_flat = _flatten_updates(updates, attr_names)
            new_pytree: T = eqx.tree_at(where_fn, pytree, updates_flat)
            return (new_pytree, *outputs)  # type: ignore[return-value]

        return impure_fn

    def decorator_ko(pure_fn: _PureFnwKwO[TwK, P, *Ts]) -> _ImpureFnwO[TwK, P, *Ts]:
        @wraps(pure_fn)
        def impure_fn(
            pytree: TwK, *args: P.args, **kwargs: P.kwargs
        ) -> tuple[TwK, *Ts]:
            next_key, sub_key = jax.random.split(pytree.key)
            updates, *outputs = pure_fn(pytree, sub_key, *args, **kwargs)
            updates = {} if updates is None else updates
            updates["key"] = next_key
            _ensure_jit(where_fn, pytree, updates, ensure_jit=ensure_jit)
            updates_flat = _flatten_updates(updates, attr_names)
            new_pytree: TwK = eqx.tree_at(where_fn, pytree, updates_flat)
            return (new_pytree, *outputs)  # type: ignore[return-value]

        del impure_fn.__wrapped__  # type: ignore[attr-defined]
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
        msg = f"{mutates.__name__} requires 'key' to be 'True' when 'where' is 'None'"
        raise ValueError(msg)

    attr_names = [] if where is None else where.split(",")
    return ["key", *attr_names] if key else attr_names


def _ensure_jit(
    where_fn: Callable[[Any], ArrayTree],
    pytree: Any,
    updates: dict[str, ArrayTree],
    *,
    ensure_jit: bool,
) -> None:
    if ensure_jit:
        _assert_is_array_tree(where_fn(pytree))
        _assert_is_array_tree(updates)


def _flatten_updates(
    updates: dict[str, ArrayTree], attr_names: list[str]
) -> list[ArrayTree]:
    return [updates[name] for name in attr_names]


def _assert_is_array_tree(tree: Any) -> None:
    if not all(eqx.is_array(leaf) for leaf in jax.tree_flatten(tree)[0]):
        msg = "Attributes changed by `mutates` must be ArrayTrees in order to use jit."
        raise TypeError(msg)
