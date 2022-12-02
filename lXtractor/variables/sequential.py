from __future__ import annotations

import typing as t
from collections import abc

from lXtractor.core.config import SeqNames
from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import SequenceVariable, MappingT, ProtFP

T = t.TypeVar('T')
V = t.TypeVar('V')

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


if __name__ == '__main__':
    raise RuntimeError
