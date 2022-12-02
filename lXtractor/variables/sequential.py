from __future__ import annotations

import typing as t
from abc import abstractmethod
from collections import abc

from more_itertools import islice_extended

from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import SequenceVariable, MappingT, ProtFP

T = t.TypeVar('T')
V = t.TypeVar('V')
K = t.TypeVar('K')

_ProtFP = ProtFP()


def try_map(p: T, m: abc.Mapping[T, V] | None):
    try:
        if m is not None:
            return m[p]
        return p
    except KeyError:
        raise FailedCalculation(f'Missing {p} in mapping')


class SeqEl(SequenceVariable):
    __slots__ = ('p', 'seq_name')

    def __init__(self, p: int, seq_name: str = SeqNames.seq1):
        self.p = p
        self.seq_name = seq_name

    @property
    def rtype(self) -> str:
        return 'str'

    def calculate(self, seq: abc.Sequence[T], mapping: t.Optional[MappingT] = None) -> T:
        p = try_map(self.p, mapping)
        try:
            return seq[p - 1]
        except IndexError:
            raise FailedCalculation(f'Missing index {p - 1} in sequence')


class PFP(SequenceVariable):
    __slots__ = ('p', 'i')

    # pfp = ProtFP()

    def __init__(self, p: int, i: int):
        self.p = p
        self.i = i

    @property
    def rtype(self) -> t.Type[float]:
        return float

    def calculate(
            self, seq: abc.Sequence[str],
            mapping: t.Optional[MappingT] = None
    ) -> float:
        p = try_map(self.p, mapping)
        try:
            return _ProtFP[(seq[p - 1], self.i)]
        except (KeyError, IndexError):
            raise FailedCalculation(f'Failed to map {p - 1} with ProtFP')


class SliceMapReduce(SequenceVariable, t.Generic[T, V, K]):
    __slots__ = ('start', 'stop', 'step', 'seq_name')

    def __init__(
            self, start: int | None = None,
            stop: int | None = None,
            step: int | None = None,
            seq_name: str = SeqNames.seq1
    ):
        self.start = start
        self.stop = stop
        self.step = step
        self.seq_name = seq_name

    @staticmethod
    @abstractmethod
    def reduce(seq: abc.Iterable[T]) -> V:
        raise NotImplementedError

    @staticmethod
    def map(seq: abc.Iterator[K]) -> abc.Iterable[T]:
        return seq

    def calculate(self, seq: abc.Iterable[K], mapping: t.Optional[MappingT] = None) -> V:
        start, stop, step = map(
            lambda x: None if x is None else try_map(x, mapping),
            [self.start, self.stop, self.step]
        )

        if start is not None:
            start -= 1

        return self.reduce(self.map(islice_extended(seq, start, stop, step)))


def make_rmr(
        reduce_fn: abc.Callable[[abc.Iterable[T]], V],
        rtype: t.Type,
        reduce_fn_name: str | None = None,
        map_fn_name: str | None = None,
        map_fn: abc.Callable[[abc.Iterator[K]], abc.Iterable[T]] | None = None
) -> t.Type[SliceMapReduce]:
    d = {
        'reduce': staticmethod(reduce_fn),
        'rtype': property(lambda _: rtype)}

    if map_fn is None:
        map_name = ''
    else:
        map_name = map_fn_name or map_fn.__name__
        d['map'] = staticmethod(map_fn)

    reduce_name = reduce_fn_name or reduce_fn.__name__

    map_name, reduce_name = map(lambda x: x.capitalize(), [map_name, reduce_name])

    return type(
        f'Slice{map_name}{reduce_name}', (SliceMapReduce,), d
    )


if __name__ == '__main__':
    raise RuntimeError
