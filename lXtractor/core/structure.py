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
from pathlib import Path

import biotite.structure as bst
import numpy as np
from more_itertools import unique_everseen
from typing_extensions import Self

import lXtractor.core.segment as lxs
from lXtractor.core.base import AminoAcidDict
from lXtractor.core.config import LigandConfig, EMPTY_STRUCTURE_ID, StructureConfig
from lXtractor.core.exceptions import NoOverlap, InitError, LengthMismatch, MissingData
from lXtractor.core.ligand import find_ligands, Ligand
from lXtractor.util.structure import (
    filter_selection,
    filter_any_polymer,
    load_structure,
    save_structure,
    filter_solvent_extended,
    filter_polymer,
)

LOGGER = logging.getLogger(__name__)
EMPTY_ALTLOC = ("", " ", ".")


class GenericStructure:
    # TODO: make truly immutable
    """
    A generic macromolecular structure with possibly many chains holding
    a single :class:`biotite.structure.AtomArray` instance.

    Methods ``__repr__`` and ``__str__`` output a string in the format:
    ``{_structure_id}:{polymer_chain_ids};{ligand_chain_ids}|{altloc_ids}``
    where ``*ids`` are ","-separated.
    """

    __slots__ = (
        "_array",
        "_structure_id",
        "_ligands",
        "_array_polymer_mask",
        "_array_solvent_mask",
        "_array_pep_mask",
        "_array_nuc_mask",
        "_array_car_mask",
        "_id",
        "_cfg",
    )

    def __init__(
        self,
        array: bst.AtomArray,
        structure_id: str,
        ligands: bool | list[Ligand] = True,
        cfg: StructureConfig = StructureConfig(),
    ):
        """
        :param array: Atom array object.
        :param structure_id: ID of a structure in `array`.
        :param ligands: A list of ligands or flag indicating to extract ligands
            during initialization.
        :param ligand_cfg: A config for ligand discovery and annotation.
        """
        #: Atom array object.
        self._array: bst.AtomArray = array
        #: ID of a structure in `array`.
        self._structure_id: str = structure_id

        self._cfg = cfg

        if isinstance(ligands, bool):
            _ligands = list(find_ligands(self)) if ligands else []
        else:
            _ligands = ligands
        #: A list of ligands
        self._ligands = _ligands

        self._array_polymer_mask = None
        self._array_solvent_mask = None
        self._array_pep_mask = None
        self._array_nuc_mask = None
        self._array_car_mask = None

        self._id = self._make_id()

    def __len__(self) -> int:
        return len(self.array)

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, GenericStructure):
            return (
                self._structure_id == other._structure_id
                and len(self) == len(other)
                and np.all(self.array == other.array)
            )
        return False

    def __hash__(self) -> int:
        atoms = tuple(
            (a.chain_id, a.res_id, a.res_name, a.atom_name, tuple(a.coord))
            for a in self.array
        )
        return hash(self._structure_id) + hash(atoms)

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def _make_id(self) -> str:
        chains_pol = ",".join(sorted(self.chain_ids_polymer))
        chains_lig = ",".join(sorted(self.chain_ids_ligand))
        altloc_ids = ",".join(filter(lambda x: x not in EMPTY_ALTLOC, self.altloc_ids))
        return f"{self._structure_id}:{chains_pol};{chains_lig}|{altloc_ids}"

    @property
    def id(self) -> str:
        return self._id

    @property
    def structure_id(self) -> str:
        return self._structure_id

    @structure_id.setter
    def structure_id(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"New ID must have the `str` type. Got {type(value)}")
        self._structure_id = value
        self._id = self._make_id()

    @property
    def cfg(self) -> StructureConfig:
        return self._cfg

    @property
    def array(self) -> bst.AtomArray:
        """
        :return: Atom array object.
        """
        return self._array

    @array.setter
    def array(self, _: t.Any) -> None:
        raise RuntimeError("Array cannot be set. Please initialize a new structure")

    @property
    def array_polymer(self) -> bst.AtomArray:
        """
        .. seealso::
            `lXtractor.util.structure.filter_any_polymer`

        :return: An atom array comprising all polymer atoms.
        """
        return self.array[self.mask_polymer & ~self.mask_solvent]

    @property
    def array_poly_peptide(self) -> bst.AtomArray:
        return self.array[self.mask_poly_peptide]

    @property
    def array_poly_nucleotide(self) -> bst.AtomArray:
        return self.array[self.mask_poly_nucleotide]

    @property
    def array_poly_carbohydrate(self) -> bst.AtomArray:
        return self.array[self.mask_poly_carbohydrate]

    @property
    def array_ligand(self) -> bst.AtomArray:
        """
        .. seealso::
            `lXtractor.util.structure.filter_ligand`

        :return: An atom array comprising all ligand atoms.
        """
        return self.array[self.mask_ligands]

    @property
    def array_solvent(self) -> bst.AtomArray:
        return self.array[self.mask_solvent]

    @property
    def altloc_ids(self) -> list[str]:
        """
        :return: A sorted list of altloc IDs. If none found, will output ``[""]``.
        """
        if hasattr(self.array, "altloc_id"):
            return sorted(set(self.array.altloc_id))
        return [""]

    @property
    def chain_ids(self) -> list[str]:
        """
        :return: A list of chain IDs this structure encompasses.
        """
        return list(unique_everseen(self.array.chain_id))

    @property
    def chain_ids_polymer(self) -> list[str]:
        """
        :return: A list of polymer chain IDs.
        """
        return list(unique_everseen(self.array_polymer.chain_id))

    @property
    def chain_ids_poly_peptide(self) -> list[str]:
        """
        :return: A list of polymeric peptide chain IDs.
        """
        return list(unique_everseen(self.array_poly_peptide.chain_id))

    @property
    def chain_ids_poly_nucleotide(self) -> list[str]:
        """
        :return: A list of polymeric nucleotide chain IDs.
        """
        return list(unique_everseen(self.array_poly_nucleotide.chain_id))

    @property
    def chain_ids_poly_carbohydrate(self) -> list[str]:
        """
        :return: A list of polymeric carbohydrate chain IDs.
        """
        return list(unique_everseen(self.array_poly_carbohydrate.chain_id))

    @property
    def chain_ids_ligand(self) -> list[str]:
        """
        :return: A set of ligand chain IDs.
        """
        return list(unique_everseen(lig.chain_id for lig in self.ligands))

    @property
    def ligands(self) -> list[Ligand]:
        """
        :return: A list of ligands.
        """
        return self._ligands

    @ligands.setter
    def ligands(self, _):
        raise RuntimeError("Cannot set ligands")

    @property
    def ligand_cfg(self) -> LigandConfig:
        return self._cfg.ligand_config

    @property
    def mask_ligands(self) -> np.ndarray:
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
    def mask_polymer(self) -> np.ndarray:
        if self._array_polymer_mask is None:
            self._array_polymer_mask = filter_any_polymer(self.array)
        return self._array_polymer_mask

    @property
    def mask_poly_peptide(self):
        if self._array_pep_mask is None:
            self._array_pep_mask = filter_polymer(self.array, pol_type="p")
        return self._array_pep_mask

    @property
    def mask_poly_nucleotide(self):
        if self._array_nuc_mask is None:
            self._array_nuc_mask = filter_polymer(self.array, pol_type="n")
        return self._array_nuc_mask

    @property
    def mask_poly_carbohydrate(self):
        if self._array_car_mask is None:
            self._array_car_mask = filter_polymer(self.array, pol_type="c")
        return self._array_car_mask

    @property
    def mask_solvent(self) -> np.ndarray:
        if self._array_solvent_mask is None:
            self._array_solvent_mask = filter_solvent_extended(self.array)
        return self._array_solvent_mask

    @property
    def mask_unclassified(self):
        return ~reduce(
            op.or_, (self.mask_polymer, self.mask_solvent, self.mask_ligands)
        )

    @property
    def is_empty(self) -> bool:
        """
        :return: ``True`` if the :meth:`array` is empty.
        """
        return len(self) == 0

    @property
    def is_empty_polymer(self) -> bool:
        """
        Check if there are any polymer atoms.

        :return: ``True`` if there are >=1 polymer atoms and ``False``
            otherwise.
        """
        return self.mask_polymer.sum() == 0

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
        path2id: abc.Callable[[Path], str] = lambda p: p.name.split(".")[0],
        structure_id: str = EMPTY_STRUCTURE_ID,
        ligands: bool = True,
        cfg: StructureConfig = StructureConfig(),
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
        :param cfg: General structure settings.
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
        if isinstance(inp, Path) and structure_id == EMPTY_STRUCTURE_ID:
            structure_id = path2id(inp)
        if isinstance(array, bst.AtomArrayStack):
            raise InitError(
                f"{inp} is likely an NMR structure. "
                f"NMR structures are not supported."
            )
        return cls(array, structure_id, ligands, cfg)

    @classmethod
    def make_empty(cls, structure_id: str = EMPTY_STRUCTURE_ID) -> Self:
        """
        :param structure_id: (Optional) ID of the created array.
        :return: An instance with empty :meth:`array`.
        """
        return cls(bst.AtomArray(0), structure_id, False)

    def write(self, path: Path) -> Path:
        """
        A one-line wrapper around :func:`lXtractor.util.structure.save_structure`.

        :param path: A path or a path-like object compatible with :func:`open`.
        :return: Path if writing was successful.
        """
        return save_structure(self.array, path)

    def rm_solvent(self, copy: bool = False):
        """
        :param copy: Copy the resulting substructure.
        :return: A substructure with solvent molecules removed.
        """
        array = self.array[~self.mask_solvent]

        if copy:
            array = array.copy()

        return self.__class__(
            array, self.structure_id, len(self.ligands) > 0, self.cfg
        )

    def get_protein_sequence(self) -> abc.Generator[tuple[str, str, int]]:
        """
        :return: A tuple with (1) one-letter code, (2) three-letter code,
            (3) residue number of each residue in :meth:`array_polymer`.
        """
        # TODO: rm specialization towards protein _seq
        if self.is_empty:
            return []

        mapping = AminoAcidDict()
        a = self.array if len(self.array_poly_peptide) == 0 else self.array_poly_peptide
        for r in bst.residue_iter(a):
            atom = r[0]
            try:
                one_letter_code = mapping.three21[atom.res_name]
            except KeyError:
                one_letter_code = "X"
            yield one_letter_code, atom.res_name, atom.res_id

    def subset_with_ligands(
        self, mask: np.ndarray, transfer_meta: bool = True, copy: bool = False
    ) -> Self:
        """
        Create a sub-structure preserving connected :attr:`ligands`.

        :param mask: Boolean mask, ``True`` for atoms in :meth:`array`, used
            to create a sub-structure.
        :param transfer_meta: Transfer a copy of existing metadata for
            connected ligands.
        :param copy: Copy the atom array resulting from subsetting the original
            one.
        :return: A new instance with atoms defined by `mask` and connected
            ligands.
        """
        # Filter connected ligands
        ligands = list(
            filter(
                lambda lig: lig.is_locally_connected(mask, self.ligand_cfg),
                self.ligands,
            )
        )
        # Extend mask by atoms from the connected ligands
        ligands_mask = reduce(
            op.or_,
            (lig.mask for lig in ligands),
            np.zeros_like(self.array.res_id, dtype=bool),
        )
        m = mask | ligands_mask
        # Create a new instance
        a = self.array[m]
        if copy:
            a = a.copy()
        new = self.__class__(a, self._structure_id, False)
        # Populate its ligands by subsetting the existing ones.
        for lig in ligands:
            meta = lig.meta.copy() if transfer_meta else None
            lig_ = Ligand(
                new,
                lig.mask[m],
                lig.contact_mask[m],
                lig.parent_contacts[m],
                lig.ligand_idx[m],
                lig.dist[m],
                meta,
            )
            new._ligands.append(lig_)

        new._id = new._make_id()

        return new

    def split_chains(
        self,
        *,
        copy: bool = False,
        polymer: bool = False,
        polymer_type: str = "all",
        ligands: bool = True,
    ) -> abc.Iterator[Self]:
        """
        Split into separate chains. Splitting is done using
        :func:`biotite.structure.get_chain_starts`.

        .. note::
            Preserved ligands may have a different ``chain_id``

        :param copy: Copy atom arrays resulting from subsetting based on
            chain annotation.
        :param polymer: Use only polymer chains for splitting.
        :param polymer_type: If `polymer` is ``True``, specify polymer type.
            Options: "all", "peptide", "nucleotide", "carbohydrate".
            Abbreviations, e.g., "a", "pep", "c", are supported.
        :param ligands: A flag indicating whether to preserve connected ligands.
        :return: An iterable over chains found in :attr:`array`.
        """
        # TODO: the chains are subsetted from copy and are not copies themselves

        if polymer:
            if polymer_type.startswith("a"):
                chain_ids = self.chain_ids_polymer
            elif polymer_type.startswith("p"):
                chain_ids = self.chain_ids_poly_peptide
            elif polymer_type.startswith("n"):
                chain_ids = self.chain_ids_poly_nucleotide
            elif polymer_type.startswith("c"):
                chain_ids = self.chain_ids_poly_carbohydrate
            else:
                raise ValueError(f"Invalid polymer type {polymer_type}")
        else:
            chain_ids = self.chain_ids

        a = self.array.copy() if copy else self.array
        for chain_id in sorted(chain_ids):
            mask = a.chain_id == chain_id
            if ligands:
                yield self.subset_with_ligands(mask, copy=copy)
            else:
                a_sub = a[mask]
                if copy:
                    a_sub = a_sub.copy()
                yield self.__class__(a_sub, self.structure_id)

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
            yield self
            return

        no_alt_mask = np.isin(self.array.altloc_id, EMPTY_ALTLOC)
        for altloc in ids[1:]:
            a = self.array[no_alt_mask | (self.array.altloc_id == altloc)]
            if copy:
                a = a.copy()
            yield self.__class__(a, self._structure_id, ligands=bool(self.ligands))

    def extract_segment(
        self,
        start: int,
        end: int,
        ligands: bool = True,
    ) -> Self:
        """
        Create a sub-structure encompassing some continuous segment bounded by
        existing position boundaries.

        :param start: Residue number to start from (inclusive).
        :param end: Residue number to stop at (inclusive).
        :param ligands: A flag indicating whether to preserve connected ligands.
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
        self_start_, self_end_, start_, end_ = map(
            lambda x: x + offset, [self_start, self_end, start, end]
        )
        if self_start_ > self_end_:
            raise NoOverlap(
                f"Invalid boundaries ({self_start}+{offset}, {self_end}+{offset}) "
                f"derived for sequence being subsetted"
            )
        if start_ > end_:
            raise NoOverlap(
                f"Invalid boundaries ({start}+{offset}, {end}+{offset}) "
                f"derived for subsetting"
            )
        seg_self = lxs.Segment(self_start_, self_end_)
        seg_sub = lxs.Segment(start_, end_)
        if not seg_self.bounds(seg_sub):
            raise NoOverlap(
                f"Provided positions {start, end} lie outside "
                f"of the structure positions {self_start, self_end}"
            )
        mask = (self.array.res_id >= start) & (self.array.res_id <= end)
        if ligands:
            return self.subset_with_ligands(mask)
        return self.__class__(self.array[mask], self._structure_id)

    def extract_positions(
        self,
        pos: abc.Sequence[int],
        chain_ids: abc.Sequence[str] | str | None = None,
    ) -> Self:
        """
        Extract specific positions from this structure.

        :param pos: A sequence of positions (res_id) to extract.
        :param chain_ids: Optionally, a single chain ID or a sequence of such.
        :return: A new instance with extracted residues.
        """

        if self.is_empty:
            return self.make_empty(self._structure_id)

        a = self.array

        mask = np.isin(a.res_id, pos)
        if chain_ids is not None:
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            mask &= np.isin(a.chain_id, chain_ids)
        if len(self.ligands) > 0:
            return self.subset_with_ligands(mask)
        return self.__class__(a[mask], self._structure_id)

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
            GenericStructure(other_transformed, other._structure_id),
            rmsd_target,
            transformation,
        )


if __name__ == "__main__":
    raise RuntimeError
