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
    Concatenate,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
)

import equinox as eqx
from jaxtyping import PyTree

T = TypeVar("T")
In = ParamSpec("In")
Out = TypeVarTuple("Out")

WhereFn: TypeAlias = Callable[[T], PyTree]
PureFn: TypeAlias = Callable[Concatenate[T, In], tuple[PyTree, *Out]]
ImpureFn: TypeAlias = Callable[Concatenate[T, In], tuple[T, *Out]]


class MutableDecorator(Protocol):
    def __call__(self, pure_fn: PureFn[T, In, *Out]) -> ImpureFn[T, In, *Out]:
        ...


def mutates(where: str | WhereFn[T]) -> MutableDecorator:
    def decorator(pure_fn: PureFn[T, In, *Out]) -> ImpureFn:
        if isinstance(where, str):
            attr_names = where.split(",")
            if len(attr_names) == 1:
                where_fn = lambda pytree: getattr(pytree, attr_names[0])  # noqa: E731
            else:
                where_fn = lambda pytree: tuple(getattr(pytree, n) for n in attr_names)  # noqa: E731
        else:
            where_fn = where

        @wraps(pure_fn)
        def impure_fn(pytree: T, *args: In.args, **kwargs: In.kwargs) -> tuple[T, *Out]:
            updated_state, *output = pure_fn(pytree, *args, **kwargs)
            updated_pytree: T = eqx.tree_at(where_fn, pytree, updated_state)
            return (updated_pytree, *output)  # type: ignore[return-value]

        return impure_fn

    return decorator
