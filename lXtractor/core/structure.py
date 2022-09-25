from __future__ import annotations

import typing as t
from collections import abc
from os import PathLike
from pathlib import Path

import biotite.structure as bst
import biotite.structure.io as strio

from lXtractor.core.base import AminoAcidDict, AbstractStructure
from lXtractor.core.exceptions import NoOverlap, InitError
from lXtractor.core.segment import Segment
from lXtractor.util.structure import read_fast_pdb


class Structure(AbstractStructure):

    def __init__(self, array: bst.AtomArray, pdb_id: t.Optional[str] = None):
        self.array = array
        self.pdb_id = pdb_id

    @classmethod
    def read(cls, path: Path) -> Structure:
        loader = read_fast_pdb if path.suffix == '.pdb' else strio.load_structure
        return cls(loader(str(path)), path.stem)

    def write(self, path: Path | PathLike | str | bytes):
        if isinstance(path, (Path, PathLike)):
            # TODO: this is temporary and should be fixed in biotite next versions
            path = str(path)
        strio.save_structure(path, self.array)

    def get_sequence(self) -> abc.Iterable[tuple[str, str, int]]:
        mapping = AminoAcidDict()
        for r in bst.residue_iter(self.array):
            atom = r[0]
            yield mapping[atom.res_name], atom.res_name, atom.res_id

    def split_chains(self, *, copy: bool = True) -> abc.Iterator[Structure]:
        chains = (self.__class__(a.copy() if copy else a, self.pdb_id)
                  for a in bst.chain_iter(self.array))
        yield from chains

    def sub_structure(self, start: int, end: int) -> Structure:
        self_start, self_end = self.array.res_id.min(), self.array.res_id.max()
        if not Segment(self_start, self_end).bounds(Segment(start, end)):
            raise NoOverlap(f'Provided positions {start, end} lie outside '
                            f'of the structure positions {self_start, self_end}')
        idx = (self.array.res_id >= start) & (self.array.res_id <= end)
        return Structure(self.array[idx], self.pdb_id)


PDB_Chain = t.NamedTuple(
    'PDB_Data', [('id', str), ('chain', str), ('structure', Structure)])


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
