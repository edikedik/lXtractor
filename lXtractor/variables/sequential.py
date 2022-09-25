from __future__ import annotations

import typing as t
from collections import abc

from lXtractor import AminoAcidDict
from lXtractor.variables.base import SequenceVariable
from lXtractor.variables.variables import _try_map, _try_find_residue


class SeqEl(SequenceVariable):
    """
    Sequence element. A residue at some alignment position.
    """

    def __init__(self, aln_pos: int):
        """
        :param aln_pos: Position at the MSA.
        """
        self.aln_pos = aln_pos
        self.amino_acid_dict = AminoAcidDict()

    @property
    def rtype(self) -> str:
        return 'str'

    def calculate(self, array: Structure = None, mapping: t.Optional[abc.Mapping[int, int]]) -> str:
        pos = _try_map(self.aln_pos, mapping)
        res = _try_find_residue(pos, array)
        resname = res.get_resname()
        return f'{pos}_{resname}_{self.amino_acid_dict[resname]}'
