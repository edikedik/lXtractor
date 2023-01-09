from __future__ import annotations

import typing as t
from collections import abc

T = t.TypeVar('T')


def is_iterable_of(
    s: abc.Iterable[t.Any], _type: t.Type[T]
) -> t.TypeGuard[abc.Iterable[T]]:
    return all(isinstance(x, _type) for x in s)


def is_sequence_of(
    s: abc.Sequence[t.Any], _type: t.Type[T]
) -> t.TypeGuard[abc.Sequence[T]]:
    return all(isinstance(x, _type) for x in s)


def is_type(x: t.Any, _type: t.Type[T]) -> t.TypeGuard[T]:
    return isinstance(x, _type)


if __name__ == '__main__':
    raise RuntimeError
