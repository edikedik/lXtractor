from __future__ import annotations

import typing as t

from lXtractor import AminoAcidDict
from lXtractor.variables.base import SequenceVariable, MappingT


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

    def calculate(self, seq: str = None, mapping: t.Optional[MappingT] = None) -> str:
        pass
        # pos = _try_map(self.aln_pos, mapping)
        # res = _try_find_residue(pos, array)
        # resname = res.get_resname()
        # return f'{pos}_{resname}_{self.amino_acid_dict[resname]}'
