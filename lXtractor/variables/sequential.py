from __future__ import annotations

import typing as t

from lXtractor.core.exceptions import FailedCalculation
from lXtractor.variables.base import SequenceVariable, MappingT


class SeqEl(SequenceVariable):
    """
    Sequence element. A residue at some alignment position.
    """

    __slots__ = ('p', )

    def __init__(self, p: int):
        """
        :param aln_pos: Position at the MSA.
        """
        self.p = p

    @property
    def rtype(self) -> str:
        return 'str'

    def calculate(self, seq: str = None, mapping: t.Optional[MappingT] = None) -> str:
        try:
            p = self.p
            if mapping is not None:
                p = mapping[self.p]
        except KeyError:
            raise FailedCalculation(f'Missing {self.p} in mapping')
        try:
            return seq[p - 1]
        except IndexError:
            raise FailedCalculation(f'Missing index {p - 1} in sequence')
