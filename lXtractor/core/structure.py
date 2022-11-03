from __future__ import annotations

import typing as t
from collections import abc
from os import PathLike
from pathlib import Path

import biotite.structure as bst
import biotite.structure.io as strio
import numpy as np

from lXtractor.core.base import AminoAcidDict, AbstractStructure
from lXtractor.core.exceptions import NoOverlap, InitError, LengthMismatch, MissingData
from lXtractor.core.segment import Segment
from lXtractor.util.structure import read_fast_pdb, filter_selection


class GenericStructure(AbstractStructure):
    __slots__ = ('array', 'pdb_id')

    def __init__(self, array: bst.AtomArray, pdb_id: t.Optional[str] = None):
        self.array = array
        self.pdb_id = pdb_id

    @classmethod
    def read(cls, path: Path) -> GenericStructure:
        loader = read_fast_pdb if path.suffix == '.pdb' else strio.load_structure
        array = loader(str(path))
        if isinstance(array, bst.AtomArrayStack):
            raise InitError(f'{path} is likely an NMR structure. '
                            f'NMR structures are not supported.')
        return cls(array, path.stem)

    def write(self, path: Path | PathLike | str | bytes):
        if isinstance(path, (Path, PathLike)):
            # TODO: this is temporary and should be fixed in biotite next versions
            path = str(path)
        strio.save_structure(path, self.array)

    def get_sequence(self) -> abc.Iterable[tuple[str, str, int]]:
        # TODO: rm specialization towards protein seq
        mapping = AminoAcidDict()
        for r in bst.residue_iter(self.array):
            atom = r[0]
            try:
                one_letter_code = mapping.three21[atom.res_name]
            except KeyError:
                one_letter_code = 'X'
            yield one_letter_code, atom.res_name, atom.res_id

    def split_chains(self, *, copy: bool = True) -> abc.Iterator[GenericStructure]:
        chains = (self.__class__(a.copy() if copy else a, self.pdb_id)
                  for a in bst.chain_iter(self.array))
        yield from chains

    def sub_structure(self, start: int, end: int) -> GenericStructure:
        self_start, self_end = self.array.res_id.min(), self.array.res_id.max()
        if not Segment(self_start, self_end).bounds(Segment(start, end)):
            raise NoOverlap(f'Provided positions {start, end} lie outside '
                            f'of the structure positions {self_start, self_end}')
        idx = (self.array.res_id >= start) & (self.array.res_id <= end)
        return GenericStructure(self.array[idx], self.pdb_id)

    def superpose(
            self, other: GenericStructure | bst.AtomArray,
            res_id_self: abc.Iterable[int] | None = None,
            res_id_other: abc.Iterable[int] | None = None,
            atom_names_self: abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None = None,
            atom_names_other: abc.Iterable[abc.Sequence[str]] | abc.Sequence[str] | None = None,
    ) -> tuple[GenericStructure, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        def _get_mask(a, res_id, atom_names):
            if res_id:
                m = filter_selection(a, res_id, atom_names)
            else:
                if atom_names:
                    m = np.isin(a.atom_name, atom_names)
                else:
                    m = np.ones_like(a, bool)
            return m

        m_self = _get_mask(self.array, res_id_self, atom_names_self)
        m_other = _get_mask(other.array, res_id_other, atom_names_other)

        num_self, num_other = m_self.sum(), m_other.sum()
        if num_self != num_other:
            raise LengthMismatch(f'To superpose, the number of atoms must match. '
                                 f'Got {num_other} in self and {num_other} in other.')

        if num_self == num_other == 0:
            raise MissingData('No atoms selected')

        superposed, transformation = bst.superimpose(self.array[m_self], other.array[m_other])
        other_transformed = bst.superimpose_apply(other.array, transformation)

        rmsd_target = bst.rmsd(self.array[m_self], superposed)

        return GenericStructure(other_transformed, other.pdb_id), rmsd_target, transformation


PDB_Chain = t.NamedTuple(
    'PDB_Chain', [('id', str), ('chain', str), ('structure', GenericStructure)])


def validate_chain(pdb: PDB_Chain):
    if pdb.structure is None:
        return
    chains = set(pdb.structure.array.chain_id)
    if len(chains) > 1:
        raise InitError('The structure must contain a single chain')
    chain_id = chains.pop()
    if chain_id != pdb.chain:
        raise InitError(f'Actual chain {chain_id} does not match .chain attribute {pdb.chain}')


if __name__ == '__main__':
    raise RuntimeError
