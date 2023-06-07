"""
Module defines basic interfaces to interact with macromolecular structures.
"""
from __future__ import annotations

import logging
import operator as op
import typing as t
from collections import abc
from functools import reduce
from io import IOBase
from os import PathLike
from pathlib import Path

import biotite.structure as bst
import biotite.structure.io as strio
import numpy as np
from typing_extensions import Self

import lXtractor.core.segment as lxs
from lXtractor.core.base import AminoAcidDict
from lXtractor.core.exceptions import NoOverlap, InitError, LengthMismatch, MissingData
from lXtractor.core.ligand import find_ligands, Ligand
from lXtractor.util.structure import (
    filter_selection,
    filter_any_polymer,
    load_structure,
)

LOGGER = logging.getLogger(__name__)
EMPTY = "Unk"
EMPTY_ALTLOC = ("", " ", ".")


class GenericStructure:
    """
    A generic macromolecular structure with possibly many chains holding
    a single :class:`biotite.structure.AtomArray` instance.

    Methods ``__repr__`` and ``__str__`` output a string in the format:
    ``{pdb_id}:{polymer_chain_ids};{ligand_chain_ids}|{altloc_ids}``
    where ``*ids`` are ","-separated.
    """

    __slots__ = ("_array", "pdb_id", "_ligands")

    def __init__(
        self,
        array: bst.AtomArray,
        pdb_id: str | None = None,
        ligands: bool | list[Ligand] = True,
    ):
        """
        :param array: Atom array object.
        :param pdb_id: PDB ID of a structure in `array`.
        :param ligands: A list of ligands or flag indicating to
        """
        #: Atom array object.
        self._array: bst.AtomArray = array
        #: PDB ID of a structure in `array`.
        self.pdb_id: str | None = pdb_id

        if isinstance(ligands, bool):
            _ligands = list(find_ligands(self)) if ligands else []
        else:
            _ligands = ligands
        #: A list of ligands
        self._ligands = _ligands

    def __len__(self) -> int:
        return len(self.array)

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, GenericStructure):
            return (
                self.pdb_id == other.pdb_id
                and len(self) == len(other)
                and np.all(self.array == other.array)
            )
        return False

    def __hash__(self) -> int:
        atoms = tuple(
            (a.chain_id, a.res_id, a.res_name, a.atom_name, tuple(a.coord))
            for a in self.array
        )
        return hash(self.pdb_id) + hash(atoms)

    def __str__(self) -> str:
        chains_pol = ",".join(sorted(self.chain_ids_polymer))
        chains_lig = ",".join(sorted(self.chain_ids_ligand))
        altloc_ids = ",".join(filter(lambda x: x not in EMPTY_ALTLOC, self.altloc_ids))
        return f"{self.pdb_id}:{chains_pol};{chains_lig}|{altloc_ids}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def array(self) -> bst.AtomArray:
        """
        :return: Atom array object.
        """
        return self._array

    @property
    def array_polymer(self) -> bst.AtomArray:
        """
        .. seealso::
            `lXtractor.util.structure.filter_any_polymer`

        :return: An atom array comprising all polymer atoms.
        """
        return self.array[filter_any_polymer(self.array)]

    @property
    def array_ligand(self) -> bst.AtomArray:
        """
        .. seealso::
            `lXtractor.util.structure.filter_ligand`

        :return: An atom array comprising all ligand atoms.
        """
        return self.array[self.ligand_mask]

    @property
    def altloc_ids(self) -> list[str]:
        """
        :return: A sorted list of altloc IDs. If none found, will output ``[""]``.
        """
        if hasattr(self.array, "altloc_id"):
            return sorted(set(self.array.altloc_id))
        return [""]

    @property
    def chain_ids(self) -> set[str]:
        """
        :return: A set of chain IDs this structure encompasses.
        """
        return set(self.array.chain_id)

    @property
    def chain_ids_polymer(self) -> set[str]:
        """
        :return: A set of non-ligand chain IDs.
        """
        return set(self.array_polymer.chain_id)

    @property
    def chain_ids_ligand(self) -> set[str]:
        """
        :return: A set of ligand chain IDs.
        """
        return {lig.chain_id for lig in self.ligands}

    @property
    def ligands(self) -> list[Ligand]:
        """
        :return: A list of ligands.
        """
        return self._ligands

    @property
    def ligand_mask(self) -> np.ndarray:
        """
        :return: A boolean mask where ``True`` points to all :meth:`ligand`
            atoms.
        """
        return reduce(
            op.or_,
            (lig.mask for lig in self.ligands),
            np.zeros_like(self.array, dtype=bool),
        )

    @property
    def is_empty(self) -> bool:
        """
        :return: ``True`` if the :meth:`array` is empty.
        """
        return len(self) == 0

    @property
    def is_singleton(self) -> bool:
        """
        :return: ``True`` if the structure contains a single residue.
        """
        return bst.get_residue_count(self.array) == 1

    @classmethod
    def read(
        cls,
        inp: IOBase | Path | str | bytes,
        path2id: abc.Callable[[Path], str] = lambda p: p.stem,
        structure_id: str = EMPTY,
        ligands: bool = True,
        altloc: bool = False,
        **kwargs,
    ) -> Self:
        """
        Parse the atom array from the provided input and wrap it into the
        :class:`GenericStructure` object.

        .. seealso::
            :func:`lXtractor.util.structure.load_structure`

        .. note::
            If `inp` is not a ``Path``, ``kwargs`` must contain the correct
            ``fmt`` (e.g., ``fmt=cif``).

        :param inp: Path to a structure in supported format.
        :param path2id: A callable obtaining a PDB ID from the file path.
            By default, it's a ``Path.stem``.
        :param structure_id: A structure unique identifier (e.g., PDB ID). If
            not provided and the input is ``Path``, will use ``path2id`` to
            infer the ID. Otherwise, will use a constant placeholder.
        :param ligands: Search for ligands.
        :param altloc: Parse alternative locations and populate
            ``array.altloc_id`` attribute.
        :param kwargs: Passed to ``load_structure``.
        :return: Parsed structure.
        """
        if altloc:
            kwargs["altloc"] = "all"
        array = load_structure(inp, **kwargs)
        if hasattr(array, "altloc_id"):
            array.altloc_id[np.isin(array.altloc_id, EMPTY_ALTLOC)] = ""
        if isinstance(inp, Path) and structure_id == EMPTY:
            structure_id = path2id(inp)
        if isinstance(array, bst.AtomArrayStack):
            raise InitError(
                f"{inp} is likely an NMR structure. "
                f"NMR structures are not supported."
            )
        return cls(array, structure_id, ligands)

    @classmethod
    def make_empty(cls, pdb_id: str | None = None) -> Self:
        """
        :param pdb_id: (Optional) PDB ID of the created array.
        :return: An instance with empty :meth:`array`.
        """
        return cls(bst.AtomArray(0), pdb_id, False)

    def write(self, path: Path | PathLike | str):
        """
        A one-line wrapper around :func:`biotite.structure.io.save_structure`.

        :param path: A path or a path-like object compatible with :func:`open`.
        :return: Nothing
        """
        strio.save_structure(path, self.array)

    def get_sequence(self) -> abc.Generator[tuple[str, str, int]]:
        """
        :return: A tuple with (1) one-letter code, (2) three-letter code,
            (3) residue number of each residue in :meth:`array_polymer`.
        """
        # TODO: rm specialization towards protein seq
        if self.is_empty:
            return []

        mapping = AminoAcidDict()
        a = self.array if len(self.array_polymer) == 0 else self.array_polymer
        for r in bst.residue_iter(a):
            atom = r[0]
            try:
                one_letter_code = mapping.three21[atom.res_name]
            except KeyError:
                one_letter_code = "X"
            yield one_letter_code, atom.res_name, atom.res_id

    def subset_with_ligands(self, mask: np.ndarray, min_connections: int = 1) -> Self:
        """
        Create a sub-structure preserving connected :attr:`ligands`.

        :param mask: Boolean mask ``True`` for atoms in :meth:`array` to create
            a sub-structure from.
        :param min_connections: Minimum number of connections required to keep
            the ligand.
        :return: A new instance with atoms defined by `mask` and connected
            ligands.
        """
        ligands = filter(
            lambda lig: lig.is_locally_connected(mask, min_connections), self.ligands
        )
        ligands_mask = reduce(
            op.or_,
            (lig.mask for lig in ligands),
            np.zeros_like(self.array.res_id, dtype=bool),
        )
        return self.__class__(self.array[mask | ligands_mask], self.pdb_id, True)

    def split_chains(
        self,
        *,
        copy: bool = False,
        polymer: bool = False,
        ligands: bool = True,
        min_connections: int = 1,
    ) -> abc.Iterator[Self]:
        """
        Split into separate chains. Splitting is done using
        :func:`biotite.structure.get_chain_starts`.

        .. note::
            Preserved ligands may have a different ``chain_id``

        :param copy: Copy atom arrays resulting from subsetting based on
            chain annotation.
        :param polymer: Use only polymer chains for splitting.
        :param ligands: A flag indicating whether to preserve connected ligands.
        :param min_connections: Minimum number of connections required to keep
            the ligand.
        :return: An iterable over chains found in :attr:`array`.
        """
        # TODO: the chains are subsetted from copy and are not copies themselves

        a = self.array.copy() if copy else self.array
        chain_ids = self.chain_ids_polymer if polymer else self.chain_ids
        for chain_id in sorted(chain_ids):
            mask = a.chain_id == chain_id
            if ligands:
                yield self.subset_with_ligands(mask, min_connections)
            else:
                yield self.__class__(a[mask], self.pdb_id)

    def split_altloc(self, *, copy: bool = True) -> abc.Iterator[Self]:
        """
        Split into substructures based on altloc IDs. Atoms missing altloc
        annotations are distributed into every substructure. Thus, even if
        the structure contains a single atom having altlocs (say, A and B),
        this method will produce two substructed identical except for this
        atom.

        If :meth:`array` does not specify any altloc ID, yields the same
        structure.

        :param copy: Copy ``AtomArray``s of substructures.
        :return: An iterator over objects of the same type initialized by
            atoms having altloc annotations.
        """
        ids = self.altloc_ids
        if len(ids) == 1:
            return self

        no_alt_mask = np.isin(self.array.altloc_id, EMPTY_ALTLOC)
        for altloc in ids[1:]:
            a = self.array[no_alt_mask | (self.array.altloc_id == altloc)]
            if copy:
                a = a.copy()
            yield self.__class__(a, self.pdb_id, ligands=bool(self.ligands))

    def extract_segment(
        self, start: int, end: int, ligands: bool = True, min_connections: int = 1
    ) -> Self:
        """
        Create a sub-structure encompassing some continuous segment bounded by
        existing position boundaries.

        :param start: Residue number to start from (inclusive).
        :param end: Residue number to stop at (inclusive).
        :param ligands: A flag indicating whether to preserve connected ligands.
        :param min_connections: Minimum number of connections required to keep
            the ligand.
        :return: A new Generic structure with residues in ``[start, end]``.
        """
        if self.is_empty:
            raise NoOverlap("Attempting to sub an empty structure")

        self_start, self_end = self.array.res_id.min(), self.array.res_id.max()

        # This is needed when some positions are <= 0 which can occur
        # in PDB structures but unsupported for a Segment.
        offset_self = abs(self_start) + 1 if self_start <= 0 else 0
        offset_start = abs(start) + 1 if start <= 0 else 0
        offset = max(offset_start, offset_self)
        seg_self = lxs.Segment(self_start + offset, self_end + offset)
        seg_sub = lxs.Segment(start + offset, end + offset)
        if not seg_self.bounds(seg_sub):
            raise NoOverlap(
                f"Provided positions {start, end} lie outside "
                f"of the structure positions {self_start, self_end}"
            )
        mask = (self.array.res_id >= start) & (self.array.res_id <= end)
        if ligands:
            return self.subset_with_ligands(mask, min_connections)
        return self.__class__(self.array[mask], self.pdb_id)

    def extract_positions(
        self,
        pos: abc.Sequence[int],
        chain_ids: abc.Sequence[str] | str | None = None,
        ligands: bool = True,
        min_connections: int = 1,
    ) -> Self:
        """
        Extract specific positions from this structure.

        :param pos: A sequence of positions (res_id) to extract.
        :param chain_ids: Optionally, a single chain ID or a sequence of such.
        :param ligands: A flag indicating whether to preserve connected ligands.
        :param min_connections: Minimum number of connections required to keep
            the ligand.
        :return: A new instance with extracted residues.
        """

        if self.is_empty:
            return self.make_empty(self.pdb_id)

        a = self.array

        mask = np.isin(a.res_id, pos)
        if chain_ids is not None:
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            mask &= np.isin(a.chain_id, chain_ids)
        if ligands:
            return self.subset_with_ligands(mask, min_connections)
        return self.__class__(a[mask], self.pdb_id)

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

        The subsetting achieved either by specifying residue numbers and atom
        names or by supplying a binary mask of the same length as the number
        of atoms in the structure.

        :param other: Other :class:`GenericStructure` or atom array.
        :param res_id_self: Residue numbers to select in this structure.
        :param res_id_other: Residue numbers to select in other structure.
        :param atom_names_self: Atom names to select in this structure given
            either per-residue or as a single sequence broadcasted to selected
            residues.
        :param atom_names_other: Same as `self`.
        :param mask_self: Binary mask to select atoms. Takes precedence over
            other selection arguments.
        :param mask_other: Same as `self`.
        :return: A tuple of (1) an `other` structure superposed onto this one,
            (2) an RMSD of the superposition, and (3) a transformation that
            had been used with :func:`biotite.structure.superimpose_apply`.
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

        if self.is_empty or other.is_empty:
            raise MissingData("Superposing empty structures is not supported")

        if mask_self is None:
            mask_self = _get_mask(self.array, res_id_self, atom_names_self)
        if mask_other is None:
            mask_other = _get_mask(other.array, res_id_other, atom_names_other)

        num_self, num_other = mask_self.sum(), mask_other.sum()
        if num_self != num_other:
            raise LengthMismatch(
                f"To superpose, the number of atoms must match. "
                f"Got {num_self} in self and {num_other} in other."
            )

        if num_self == num_other == 0:
            raise MissingData("No atoms selected")

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


if __name__ == "__main__":
    raise RuntimeError
