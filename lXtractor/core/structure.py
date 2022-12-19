"""
Module defines basic interfaces to interact with macromolecular structures.
"""
from __future__ import annotations

import typing as t
from collections import abc
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import biotite.structure as bst
import biotite.structure.io as strio
import numpy as np

from lXtractor.core.base import AminoAcidDict, AbstractStructure
from lXtractor.core.exceptions import NoOverlap, InitError, LengthMismatch, MissingData
from lXtractor.core.segment import Segment
from lXtractor.util.structure import filter_selection


class GenericStructure(AbstractStructure):
    """
    A generic macromolecular structure with possibly many chains holding
    a single :class:`biotite.structure.AtomArray` instance.
    """

    __slots__ = ('array', 'pdb_id')

    def __init__(self, array: bst.AtomArray, pdb_id: t.Optional[str] = None):
        """
        :param array: Atom array object.
        :param pdb_id: PDB ID of a structure in `array`.
        """
        #: Atom array object.
        self.array: bst.AtomArray = array
        #: PDB ID of a structure in `array`.
        self.pdb_id: str = pdb_id

    @classmethod
    def read(cls, path: Path) -> GenericStructure:
        array = strio.load_structure(str(path))
        if isinstance(array, bst.AtomArrayStack):
            raise InitError(
                f'{path} is likely an NMR structure. '
                f'NMR structures are not supported.'
            )
        return cls(array, path.stem)

    def write(self, path: Path | PathLike | str) -> t.NoReturn:
        """
        A one-line wrapper around :func:`biotite.structure.io.save_structure`.

        :param path: A path or a path-like object compatible with :func:`open`.
        :return: Nothing
        """
        strio.save_structure(path, self.array)

    def get_sequence(self) -> abc.Iterable[tuple[str, str, int]]:
        """
        :return: A tuple with (1) one-letter code, (2) three-letter code,
            (3) residue number.
        """
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
        """
        Split into separate chains. Splitting is done using
        :func:`biotite.structure.get_chain_starts`.

        :param copy: Copy atom arrays resulting from subsetting based on
            chain annotation.
        :return: An iterable over chains found in :attr:`array`.
        """
        chains = (
            self.__class__(a.copy() if copy else a, self.pdb_id)
            for a in bst.chain_iter(self.array)
        )
        yield from chains

    def sub_structure(self, start: int, end: int) -> GenericStructure:
        """
        Create a sub-structure encompassing some continuout segment.

        :param start: Residue number to start from (inclusive).
        :param end: Residue number to stop at (inclusive).
        :return: A new Generic structure with residues in ``[start, end]``.
        """
        self_start, self_end = self.array.res_id.min(), self.array.res_id.max()
        if not Segment(self_start, self_end).bounds(Segment(start, end)):
            raise NoOverlap(
                f'Provided positions {start, end} lie outside '
                f'of the structure positions {self_start, self_end}'
            )
        idx = (self.array.res_id >= start) & (self.array.res_id <= end)
        return GenericStructure(self.array[idx], self.pdb_id)

    def superpose(
        self,
        other: GenericStructure | bst.AtomArray,
        res_id_self: abc.Iterable[int] | None = None,
        res_id_other: abc.Iterable[int] | None = None,
        atom_names_self: abc.Iterable[abc.Sequence[str]]
        | abc.Sequence[str]
        | None = None,
        atom_names_other: abc.Iterable[abc.Sequence[str]]
        | abc.Sequence[str]
        | None = None,
        mask_self: np.ndarray | None = None,
        mask_other: np.ndarray | None = None,
    ) -> tuple[GenericStructure, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Superpose other structure to this one.
        Arguments to this function all serve a single purpose: to correctly
        subset both structures so the resulting selections have the same number
        of atoms.

        The subsetting achieved either by speficiying residue numbers and atom
        names or by suppliying a binary mask of the same length as the number
        of atoms in the structure.

        :param other: Other :class:`GenericStructure` or atom array.
        :param res_id_self: Residue numbers to select in this structure.
        :param res_id_other: Residue numbers to select in other structure.
        :param atom_names_self: Atom names to select in this structure given
            either per-residue or broadcasted to selected residues.
        :param atom_names_other: Same as `self`.
        :param mask_self: Binary mask to select atoms. Takes precedence over
            other selection arguments.
        :param mask_other: Same as `self`.
        :return:
        """

        def _get_mask(a, res_id, atom_names):
            if res_id:
                m = filter_selection(a, res_id, atom_names)
            else:
                if atom_names:
                    m = np.isin(a.atom_name, atom_names)
                else:
                    m = np.ones_like(a, bool)
            return m

        if mask_self is None:
            mask_self = _get_mask(self.array, res_id_self, atom_names_self)
        if mask_other is None:
            mask_other = _get_mask(other.array, res_id_other, atom_names_other)

        num_self, num_other = mask_self.sum(), mask_other.sum()
        if num_self != num_other:
            raise LengthMismatch(
                f'To superpose, the number of atoms must match. '
                f'Got {num_self} in self and {num_other} in other.'
            )

        if num_self == num_other == 0:
            raise MissingData('No atoms selected')

        superposed, transformation = bst.superimpose(
            self.array[mask_self], other.array[mask_other]
        )
        other_transformed = bst.superimpose_apply(other.array, transformation)

        rmsd_target = bst.rmsd(self.array[mask_self], superposed)

        return (
            GenericStructure(other_transformed, other.pdb_id),
            rmsd_target,
            transformation,
        )


@dataclass
class PDB_Chain:
    """
    A container to hold the data of a single structural chain.
    """
    id: str
    chain: str
    structure: GenericStructure


def _validate_chain(pdb: PDB_Chain):
    if pdb.structure is None:
        return
    chains = set(pdb.structure.array.chain_id)
    if len(chains) > 1:
        raise InitError('The structure must contain a single chain')
    chain_id = chains.pop()
    if chain_id != pdb.chain:
        raise InitError(
            f'Actual chain {chain_id} does not match .chain attribute {pdb.chain}'
        )


if __name__ == '__main__':
    raise RuntimeError
