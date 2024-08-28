"""
Module defines basic interfaces to interact with macromolecular structures.
"""
from __future__ import annotations

import logging
import operator as op
import typing as t
from collections import abc, defaultdict
from dataclasses import dataclass
from functools import reduce
from io import IOBase
from os import PathLike
from pathlib import Path

import biotite.structure as bst
import numpy as np
import numpy.typing as npt
import rustworkx as rx
from more_itertools import unique_everseen
from toolz import keyfilter

import lXtractor.core.segment as lxs
import lXtractor.util as util
from lXtractor.core.base import ResNameDict
from lXtractor.core.config import AtomMark, DefaultConfig, EMPTY_ALTLOC
from lXtractor.core.exceptions import NoOverlap, InitError, LengthMismatch, MissingData
from lXtractor.core.ligand import Ligand, make_ligand, ligands_from_atom_marks


LOGGER = logging.getLogger(__name__)
RES_DICT = ResNameDict()
_POL_MARKS = {
    "c": AtomMark.CARB,
    "n": AtomMark.NUC,
    "p": AtomMark.PEP,
    "x": AtomMark.UNK,
}


@dataclass(frozen=True)
class Masks:
    primary_polymer: npt.NDArray[np.bool_]
    primary_polymer_ptm: npt.NDArray[np.bool_]
    primary_polymer_modified: npt.NDArray[np.bool_]
    solvent: npt.NDArray[np.bool_]
    ligand: npt.NDArray[np.bool_]
    ligand_covalent: npt.NDArray[np.bool_]
    ligand_poly: npt.NDArray[np.bool_]
    ligand_nonpoly: npt.NDArray[np.bool_]
    ligand_pep: npt.NDArray[np.bool_]
    ligand_nuc: npt.NDArray[np.bool_]
    ligand_carb: npt.NDArray[np.bool_]
    unk: npt.NDArray[np.bool_]


class GenericStructure:
    """
    A generic macromolecular structure with possibly many chains holding
    a single :class:`biotite.structure.AtomArray` instance.

    This object is a core data structure in `lXtractor` for structural data.

    The object is considered immutable: atoms of a structure can't change their
    location or properties, as well as other protected attributes.

    While atoms are stored as :class:`biotite.structure.AtomArray`,
    `GenericStructure` defines additional annotations for each atom and
    operations crucial for other objects such as
    :class:`lXtractor.core.chain.ChainStructure`.

    Upon initialization, atom array attains graph representation (:meth:`graph`)
    using :func:`lXtractor.util.structure.to_graph` function. Using this
    representation, atom annotations are attained via :func``mark_atoms_g`.
    These annotations can be accessed via :meth:`atom_marks`. For convenience,
    boolean masks are stored and can be applied to the :meth:`array` as follows:

    .. code-block:: python

        # Assume ``s`` is a :class:`GenericStructure` object.
        s[s.mask.`mask_name`]

    To view available mask names, see :class:`Masks`.

    One of the most crucial annotations is the so-called "primary_polymer".
    These atoms serve as a frame of reference for all other atoms in a structure.
    The rest of the atoms are categorized as either ligand or solvent. Sometimes
    the annotation process fails to identify certain atoms. In such cases, a
    warning is logged. To view uncategorized atoms, one can use the following mask:

    .. code-block:: python

        s[s.mask.unk]

    .. note::
        Using ``__getitem__(item)`` like in ``s[s.mask.unk`` will return an
        atom array. Use :meth:`subset` to obtain a new generic structure or
        initialize a new ``GenericStructure(s[s.mask.unk] instance; it will be
        equivalent.

    Methods ``__repr__`` and ``__str__`` output a string in the format:
    ``{_name}:{polymer_chain_ids};{ligand_chain_ids}|{altloc_ids}``
    where ``*ids`` are ","-separated.
    """

    __slots__ = (
        "_atom_marks",
        "_array",
        "_graph",
        "_name",
        "_ligands",
        "_mask",
        "_id",
    )

    def __init__(
        self,
        array: bst.AtomArray,
        name: str,
        ligands: abc.Sequence[Ligand] | None = None,
        atom_marks: npt.NDArray[int] | PathLike | None = None,
        graph: rx.PyGraph | dict | PathLike | None = None,
    ):
        """
        :param array: Atom array object.
        :param name: ID of a structure in `array`.
        :param ligands: A list of ligands or flag indicating to extract ligands
            during initialization.
        """
        #: Atom array object.
        self._array: bst.AtomArray = array
        #: ID of a structure in `array`.
        self._name: str = name

        if isinstance(graph, rx.PyGraph):
            self._graph = graph
        elif isinstance(graph, dict | PathLike):
            self._graph = util.json_to_molgraph(graph)
        else:
            self._graph = util.to_graph(array, True)

        if len(self._graph) != len(self._array):
            raise LengthMismatch(
                f"The number of nodes in a graph ({len(self._graph)}) does not match "
                f"the number of atoms in the array ({len(self._array)})."
            )

        if atom_marks is None:
            atom_marks, primary_pol_type, _ligands = mark_atoms_g(self)
            atom_marks.flags.writeable = False
            self._atom_marks = atom_marks

            if isinstance(ligands, list):
                _ligands = ligands
        else:
            if isinstance(atom_marks, PathLike):
                atom_marks = np.load(atom_marks)
            if not isinstance(atom_marks, np.ndarray):
                raise TypeError(
                    f"Expected `atom_marks` to be an array, got {type(atom_marks)}"
                )
            if len(atom_marks) != len(array):
                raise LengthMismatch(
                    "The lengths of `atom_marks` and `array` must match. "
                    f"Got {len(atom_marks)} and {len(array)}."
                )
            self._atom_marks = atom_marks
            # determine primary polymer type
            primary_pol_type = util.find_first_polymer_type(atom_marks)
            if not isinstance(ligands, list):
                _ligands = list(ligands_from_atom_marks(self))
            else:
                _ligands = ligands

        if any(not isinstance(x, Ligand) for x in _ligands):
            raise TypeError(
                "Some entries in supplied `ligands` are not of the `Ligand` type"
            )
        #: A tuple of ligands
        self._ligands: tuple[Ligand, ...] = tuple(
            sorted(_ligands, key=lambda x: (x.res_name, x.res_id, x.chain_id))
        )

        if np.any(atom_marks == AtomMark.UNK):
            num_unk = np.sum(atom_marks == AtomMark.UNK)
            LOGGER.warning(f"Structure {name} has {num_unk} uncategorized atoms.")

        ligand_pep = atom_marks == (AtomMark.PEP | AtomMark.LIGAND)
        ligand_nuc = atom_marks == (AtomMark.NUC | AtomMark.LIGAND)
        ligand_carb = atom_marks == (AtomMark.CARB | AtomMark.LIGAND)
        ligand_poly = ligand_pep | ligand_nuc | ligand_carb
        ligand_nonpoly = atom_marks == AtomMark.LIGAND
        ligand_covalent = atom_marks == (AtomMark.LIGAND | AtomMark.COVALENT)

        if primary_pol_type == "x":
            primary_polymer = np.full_like(atom_marks, False, bool)
        else:
            primary_polymer = atom_marks == _POL_MARKS[primary_pol_type]

        primary_polymer_ptm = primary_polymer & array.hetero
        primary_polymer_mod = (
            atom_marks == (AtomMark.SOLVENT | AtomMark.COVALENT)
        ) | primary_polymer_ptm

        self._mask = Masks(
            primary_polymer=primary_polymer,
            primary_polymer_ptm=primary_polymer_ptm,
            primary_polymer_modified=primary_polymer_mod,
            solvent=(atom_marks == AtomMark.SOLVENT),
            ligand=ligand_nonpoly | ligand_poly,
            ligand_covalent=ligand_covalent,
            ligand_nonpoly=ligand_nonpoly,
            ligand_poly=ligand_poly,
            ligand_pep=ligand_pep,
            ligand_nuc=ligand_nuc,
            ligand_carb=ligand_carb,
            unk=(atom_marks == AtomMark.UNK),
        )

        self._id = self._make_id()

    def __len__(self) -> int:
        return len(self.array)

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, GenericStructure):
            return (
                    self._name == other._name
                    and len(self) == len(other)
                    and np.all(self.atom_marks == other.atom_marks)
                    and util.compare_coord(self.array, other.array)
                    and self.ligands == other.ligands
            )
        return False

    def __hash__(self) -> int:
        atoms = tuple(
            (a.chain_id, a.res_id, a.res_name, a.atom_name, tuple(a.coord))
            for a in self.array
        )
        return hash(self._name) + hash(atoms)

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def __getitem__(self, item: t.Any) -> bst.AtomArray:
        return self._array.__getitem__(item)

    def _make_id(self) -> str:
        chains_pol = ",".join(sorted(self.chain_ids_polymer))
        chains_lig = ",".join(sorted(self.chain_ids_ligand))
        altloc_ids = ",".join(filter(lambda x: x not in EMPTY_ALTLOC, self.altloc_ids))
        return f"{self._name}:{chains_pol};{chains_lig}|{altloc_ids}"

    @property
    def atom_marks(self) -> npt.NDArray[np.int_]:
        """
        :return: An array of :class:`lXtractor.core.config.AtomMark` marks,
            categorizing each atom in this structure.
        """
        return self._atom_marks

    @property
    def graph(self) -> rx.PyGraph:
        """
        :return: A structure's graph representation.
        """
        return self._graph

    @property
    def id(self) -> str:
        """
        :return: An identifier of this structure. It's composed once upon
            initialization and has the following format:
            ``{_name}:{polymer_chain_ids};{ligand_chain_ids}|{altloc_ids}``.
            It should uniquely identify a structure, i.e., one should expect
            two structures with the same ID to be identical.
        """
        return self._id

    @property
    def name(self) -> str:
        """
        :return: A name of the structure.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"New ID must have the `str` type. Got {type(value)}")
        self._name = value
        self._id = self._make_id()

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
    def mask(self) -> Masks:
        return self._mask

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
        return list(unique_everseen(self.array[self.mask.primary_polymer].chain_id))

    @property
    def chain_ids_ligand(self) -> list[str]:
        """
        :return: A set of ligand chain IDs.
        """
        return list(unique_everseen(lig.chain_id for lig in self.ligands))

    @property
    def ligands(self) -> tuple[Ligand, ...]:
        """
        :return: A list of ligands.
        """
        return self._ligands

    @ligands.setter
    def ligands(self, _):
        raise RuntimeError("Cannot set ligands")

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
        return self.mask.primary_polymer.sum() == 0

    @property
    def is_singleton(self) -> bool:
        """
        :return: ``True`` if the structure contains a single residue.
        """
        return bst.get_residue_count(self.array) == 1

    @classmethod
    def make_empty(
        cls, structure_id: str = DefaultConfig["unknowns"]["structure_id"]
    ) -> t.Self:
        """
        :param structure_id: (Optional) ID of the created array.
        :return: An instance with empty :meth:`array`.
        """
        return cls(bst.AtomArray(0), structure_id, [])

    @classmethod
    def read(
        cls,
        inp: IOBase | Path | str | bytes,
        path2id: abc.Callable[[Path], str] = lambda p: p.name.split(".")[0],
        structure_id: str = DefaultConfig["unknowns"]["structure_id"],
        altloc: bool | str = False,
        **kwargs,
    ) -> t.Self:
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
        :param altloc: Parse alternative locations and populate
            ``array.altloc_id`` attribute.
        :param kwargs: Passed to ``load_structure``.
        :return: Parsed structure.
        """
        if isinstance(altloc, bool) and altloc:
            altloc = "all"
        if isinstance(altloc, str):
            kwargs["altloc"] = altloc
        array = util.load_structure(inp, **kwargs)
        empty_id = DefaultConfig["unknowns"]["structure_id"]
        if hasattr(array, "altloc_id"):
            array.altloc_id[np.isin(array.altloc_id, EMPTY_ALTLOC)] = ""
        if isinstance(inp, Path) and structure_id == empty_id:
            structure_id = path2id(inp)
        if isinstance(array, bst.AtomArrayStack):
            raise InitError(
                f"{inp} is likely an NMR structure. "
                f"NMR structures are not supported."
            )
        if isinstance(inp, Path):
            files = util.get_files(inp.parent)
            name = inp.stem.split(".")[0]
            marks_name, graph_name = f"{name}.npy", f"{name}.json"
            marks = files.get(marks_name, None)
            graph = files.get(graph_name, None)
        else:
            marks, graph = None, None

        return cls(array, structure_id, ligands=None, atom_marks=marks, graph=graph)

    def write(
        self, path: PathLike | str, atom_marks: bool = True, graph: bool = True
    ) -> Path:
        """
        Save this structure to a file. The format is automatically determined
        from the given path.

        Additional files are saved using the same filename alongside the
        structure file. The filename will resolve to "structure" in all the
        following cases and result in "structure.npy" and "structure.json"
        files saved to the same dir::

            path="/path/to/structure.pdb"
            path="/path/to/structure.mmtf.gz"
            path="/path/to/structure.with.many.dots.pdb.gz"

        .. seealso::
            :func:`lXtractor.util.structure.save_structure`.

        :param path: A path or a path-like object compatible with :func:`open`.
            Must not point to an existing directory. Must provide the structure
            format as an extension.
        :param atom_marks: Save an array of atom marks in the `npy` format.
        :param graph: Save molecular connectivity graph in the `json` format.
        :return: Path to the saved structure if writing was successful.
        """
        if not isinstance(path, Path):
            path = Path(path)
        saved_path = util.save_structure(self.array, path)
        stem = path.stem.split(".")[0]
        if atom_marks:
            np.save(path.parent / f"{stem}.npy", self.atom_marks)
        if graph:
            json_path = path.parent / f"{stem}.json"
            util.molgraph_to_json(self.graph, path=json_path)
        return saved_path

    def get_sequence(self) -> abc.Generator[tuple[str, str, int]]:
        """
        :return: A generator over tuples, where each residue is described by:
            (1) one-letter code, (2) three-letter code, (3) residue number.
        """
        if self.is_empty:
            return []

        m = self.mask.primary_polymer
        a = self.array if not np.any(m) else self.array[m]
        first_atoms = (r[0] for r in bst.residue_iter(a))
        for i, a in enumerate(first_atoms, start=1):
            try:
                res3 = RES_DICT[a.res_name]
            except KeyError:
                LOGGER.warning(
                    f"Failed to obtain a single-letter code for residue {a.res_name} "
                    f"at {i}-th position ({a.res_id + 1}-th first atom position in "
                    f"protein) for a structure {self}."
                )
                res3 = "X"
            yield res3, a.res_name, a.res_id

    def subset(
        self,
        mask: np.ndarray,
        ligands: bool = True,
        reinit_ligands: bool = False,
        copy: bool = False,
    ) -> t.Self:
        """
        Create a sub-structure potentially preserving connected :meth:`ligands`.

        .. warning::

            If ``DefaultConfig["structure"]["primary_pol_type"]`` is set to auto,
            and `mask` points to a polymer that is shorter than some existing
            ligand polymer, this ligand polymer will become a primary polymer
            in the substructure.

        :param mask: Boolean mask, ``True`` for atoms in :meth:`array`, used
            to create a sub-structure.
        :param ligands: Keep ligands that are connected to atoms specified by
            `mask`.
        :param reinit_ligands: Reinitialize ligands upon creating a sub-structure,
            rather than filtering existing ligands connected to atoms specified
            by `mask`. Takes precedence over the `ligands` option. This option
            is used in :meth:`split_altloc`.
        :param copy: Copy the atom array resulting from subsetting the original
            one.
        :return: A new instance with atoms defined by `mask` and connected
            ligands.
        """

        if reinit_ligands:
            ligands = False

        if ligands:
            # Filter connected ligands
            ligands = list(
                filter(
                    lambda lig: lig.is_locally_connected(mask),
                    self.ligands,
                )
            )
            # Extend mask by atoms from the connected ligands
            ligand_mask = reduce(
                op.or_,
                (lig.mask for lig in ligands),
                np.zeros_like(self.array.res_id, dtype=bool),
            )
        else:
            ligand_mask = mask

        _mask = mask | ligand_mask
        a = self.array[_mask]
        g = util.graph_reindex_nodes(self.graph.subgraph(np.where(_mask)[0]))
        m = self.atom_marks[_mask]
        if copy:
            a = a.copy()
            m = m.copy()

        s = self.__class__(
            a, self.name, ligands=True if reinit_ligands else [], atom_marks=m, graph=g
        )

        if ligands:
            new_ligands = []
            names = DefaultConfig["metadata"]
            retain_names = [
                names["res_name"],
                names["res_id"],
                names["structure_chain_id"],
            ]
            for lig in ligands:
                meta = keyfilter(lambda x: x in retain_names, lig.meta)
                new_ligands.append(
                    Ligand(
                        s,
                        lig.mask[_mask],
                        lig.contact_mask[_mask],
                        lig.ligand_idx[_mask],
                        lig.dist[_mask],
                        meta,
                    )
                )
            s._ligands = tuple(
                sorted(new_ligands, key=lambda x: (x.res_name, x.res_id, x.chain_id))
            )
            s._id = s._make_id()

        return s

    def rm_solvent(self, copy: bool = False):
        """
        :param copy: Copy the resulting substructure.
        :return: A substructure with solvent molecules removed.
        """
        return self.subset(~self.mask.solvent, ligands=True, copy=copy)

    def split_chains(self, polymer: bool = False, **kwargs) -> abc.Iterator[t.Self]:
        """
        Split into separate chains. Splitting is done using
        :func:`biotite.structure.get_chain_starts`.

        .. note::
            Preserved ligands may have a different ``chain_id``.

        .. note::
            If there is a single chain, this method will return ``self``.

        :param polymer: Use only primary polymer chains for splitting.
        :param kwargs: Passed to :meth:`subset`.
        :return: An iterable over chains found in :attr:`array`.
        """

        if polymer:
            chain_ids = self.chain_ids_polymer
        else:
            chain_ids = self.chain_ids

        # a = self.array.copy() if copy else self.array
        a = self.array
        if len(chain_ids) == 1:
            yield self
            return

        for chain_id in sorted(chain_ids):
            mask = a.chain_id == chain_id
            yield self.subset(mask, **kwargs)

    def split_altloc(self, **kwargs) -> abc.Iterator[t.Self]:
        """
        Split into substructures based on altloc IDs. Atoms missing altloc
        annotations are distributed into every substructure. Thus, even if
        a structure contains a single atom having altlocs (say, A and B),
        this method will produce two substructed identical except for this
        atom.

        .. note::
            If :meth:`array` does not specify any altloc ID, the method yields
            ``self``.

        :param kwargs: Passed to :meth:`subset`.
        :return: An iterator over objects of the same type initialized by
            atoms having altloc annotations.
        """
        ids = self.altloc_ids
        if len(ids) == 1:
            yield self
            return

        if "reinit_ligands" not in kwargs:
            kwargs["reinit_ligands"] = True

        no_alt_mask = np.isin(self.array.altloc_id, EMPTY_ALTLOC)
        for altloc in ids[1:]:
            m = no_alt_mask | (self.array.altloc_id == altloc)
            yield self.subset(m, **kwargs)

    def extract_segment(self, start: int, end: int, chain_id: str, **kwargs) -> t.Self:
        """
        Create a sub-structure encompassing some continuous segment bounded by
        existing position boundaries.

        :param start: Residue number to start from (inclusive).
        :param end: Residue number to stop at (inclusive).
        :param chain_id: Chain to extract a segment from.
        :param kwargs: Passed to :meth:`subset`.
        :return: A new Generic structure with residues in ``[start, end]``.
        """
        if self.is_empty:
            raise NoOverlap("Attempting to sub an empty structure")

        chain_mask = self.array.chain_id == chain_id
        a = self.array[chain_mask]
        self_start, self_end = a.res_id.min(), a.res_id.max()

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
        mask = chain_mask & (self.array.res_id >= start) & (self.array.res_id <= end)
        return self.subset(mask, **kwargs)

    def extract_positions(
        self,
        pos: abc.Sequence[int],
        chain_ids: abc.Sequence[str] | str | None = None,
        **kwargs,
    ) -> t.Self:
        """
        Extract specific positions from this structure.

        :param pos: A sequence of positions (res_id) to extract.
        :param chain_ids: Optionally, a single chain ID or a sequence of such.
        :param kwargs: Passed to :meth:`subset`.
        :return: A new instance with extracted residues.
        """

        if self.is_empty:
            return self.make_empty(self._name)

        a = self.array

        mask = np.isin(a.res_id, pos)
        if chain_ids is not None:
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            mask &= np.isin(a.chain_id, chain_ids)
        return self.subset(mask, **kwargs)

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
                m = util.filter_selection(a, res_id, atom_names)
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
            GenericStructure(other_transformed, other._name),
            rmsd_target,
            transformation,
        )


class ProteinStructure(GenericStructure):
    """
    A structure type where primary polymer is peptide.

    .. seealso::
        :class:`GenericStructure` for general-purpose documentation.
    """

    def __init__(
        self,
        array: bst.AtomArray,
        structure_id: str,
        ligands: bool | list[Ligand] = True,
        atom_marks: npt.NDArray[int] | PathLike | None = None,
        graph: rx.PyGraph | dict | PathLike | None = None,
    ):
        with DefaultConfig.temporary_namespace():
            DefaultConfig["structure"]["primary_pol_type"] = "p"
            super().__init__(array, structure_id, ligands, atom_marks, graph)


class NucleotideStructure(GenericStructure):
    """
    A structure type where primary polymer is nucleotide.

    .. seealso::
        :class:`GenericStructure` for general-purpose documentation.
    """

    def __init__(
        self,
        array: bst.AtomArray,
        structure_id: str,
        ligands: bool | list[Ligand] = True,
        atom_marks: npt.NDArray[int] | PathLike | None = None,
        graph: rx.PyGraph | dict | PathLike | None = None,
    ):
        with DefaultConfig.temporary_namespace():
            DefaultConfig["structure"]["primary_pol_type"] = "n"
            super().__init__(array, structure_id, ligands, atom_marks, graph)


class CarbohydrateStructure(GenericStructure):
    """
    A structure type where primary polymer is carbohydrate.

    .. seealso::
        :class:`GenericStructure` for general-purpose documentation.
    """

    def __init__(
        self,
        array: bst.AtomArray,
        structure_id: str,
        ligands: bool | list[Ligand] = True,
        atom_marks: npt.NDArray[int] | PathLike | None = None,
        graph: rx.PyGraph | dict | PathLike | None = None,
    ):
        with DefaultConfig.temporary_namespace():
            DefaultConfig["structure"]["primary_pol_type"] = "c"
            super().__init__(array, structure_id, ligands, atom_marks, graph)


def mark_atoms(
    structure: GenericStructure,
) -> tuple[npt.NDArray[np.int_], list[Ligand]]:
    """
    Mark each atom in structure according to
    :class:`lXtractor.core.config.AtomMark`.

    This function is used upon initializing :class:`GenericStructure` and its
    subclasses, storing the output under :attr:`GenericStructure.atom_marks`.

    :param structure: An arbitrary structure.
    :return: An array of atom marks (equivalently, classes or types).
    """
    a = structure.array

    is_solv = util.filter_solvent_extended(a)
    pol_types = util.mark_polymer_type(a, DefaultConfig["structure"]["n_monomers"])
    is_nuc, is_pep, is_carb = (pol_types == p for p in ["n", "p", "c"])

    is_any_pol = pol_types != "x"
    is_solv[is_any_pol] = False

    marks = np.full(len(a), AtomMark.UNK)
    marks[is_solv] = AtomMark.SOLVENT

    match DefaultConfig["structure"]["primary_pol_type"][0]:
        case "p":
            is_pol, pol_type = is_pep, "p"
        case "n":
            is_pol, pol_type = is_nuc, "n"
        case "c":
            is_pol, pol_type = is_carb, "c"
        case _:
            is_pol, pol_type = max(
                [(is_carb, "c"), (is_nuc, "n"), (is_pep, "p")], key=lambda x: x[0].sum()
            )

    if not any(is_pol):
        LOGGER.warning("No polymer atoms identified. Marking as SOLVENT+UNK.")
        return marks, []

    marks[is_carb] = AtomMark.CARB
    marks[is_nuc] = AtomMark.NUC
    marks[is_pep] = AtomMark.PEP

    ligands = []

    # Annotate small-molecule ligands
    is_putative_lig = ~(is_any_pol | is_solv)
    if np.any(is_putative_lig):
        for m_res in util.iter_residue_masks(a):
            # A mask that is a single ligand residue
            m_lig = is_putative_lig & m_res
            lig = make_ligand(m_lig, is_pol, structure)
            if lig is not None:
                marks[m_lig] = AtomMark.LIGAND
                ligands.append(lig)

    # Annotate polymer ligands
    is_putative_lig = ~(is_pol | is_solv | (marks == AtomMark.LIGAND)) & is_any_pol
    pol_marks = {"c": AtomMark.CARB, "n": AtomMark.NUC, "p": AtomMark.PEP}
    if np.any(is_putative_lig):
        for c in bst.get_chains(a[is_putative_lig]):
            m_lig = is_putative_lig & (a.chain_id == c)
            lig = make_ligand(m_lig, is_pol, structure)
            if (
                lig is not None
                and lig.res_name[0] in DefaultConfig["structure"]["ligand_pol_types"]
            ):
                marks[m_lig] = AtomMark.LIGAND | pol_marks[lig.res_name[0]]
                ligands.append(lig)

    return marks, ligands


def _get_one_chain(a: bst.AtomArray) -> str:
    chain_ids = set(a.chain_id)
    if len(chain_ids) != 1:
        raise ValueError(f"Expected single chain, found {chain_ids}")
    return chain_ids.pop()


def _to_single_poly_mask(
    a: bst.AtomArray,
    poly_masks: abc.Sequence[npt.NDArray[np.bool_]],
):
    largest_idx = np.argmax([x.sum() for x in poly_masks])
    pol_mask = poly_masks[largest_idx]
    pol_chain = _get_one_chain(a[pol_mask])

    for i, m in enumerate(poly_masks):
        if i == largest_idx:
            continue
        a_sub = a[m]
        # If chains match and not all atoms are hetero
        if _get_one_chain(a_sub) == pol_chain and np.any(~a_sub.hetero):
            pol_mask[m] = True

    return pol_mask


def _combine_poly_masks(
    a: bst.AtomArray,
    poly_masks: abc.Sequence[npt.NDArray[np.bool_]],
):
    poly_mask = np.full_like(a, False, dtype=np.bool_)
    for m in poly_masks:
        if np.sum(a[m].hetero) < m.sum() / 2:
            poly_mask[m] = True

    return poly_mask


def mark_atoms_g(
    s: GenericStructure, single_poly_chain: bool = False
) -> (npt.NDArray[np.int_], str, list[Ligand]):
    """
    Mark structure atoms based on a molecular graph's representation by of
    the :class:`lXtractor.core.config.AtomMark` categories.

    Atoms are classified into five categories::

        #. primary polymer: corresponds to ``PEP``, ``NUC`` or ``CARB``
        categories.
        #. solvent: ``SOLVENT``.
        #. non polymer ligand: ``LIGAND``.
        #. polymer ligand: A combination of ``LIGAND`` with one of the primary
        polymer types, eg. ``AtomMark.LIGAND | AtomMark.NUC``.
        #. unknown: ``UNK`` for atoms that couldn't be categorized.

    The classification process depends on groups of atoms forming covalent bonds
    with each other, or connected components in the molecular graph representation.
    Each such component is assessed separately and its atoms are classified
    as polymer, ligand, or solvent. If the primary polymer is set to "auto" in
    config (``DefaultConfig["structure"]["primary_pol_type"]``), the polymer
    with the largest number of monomers will be selected. The rest of the polymers
    will become polymer ligands: special kind of ligand that can have multiple
    residues. See :class:`lXtractore.core.ligand.Ligand` for details.

    :param s:
    :param single_poly_chain:
    :return:
    """
    a = s.array
    g = s.graph
    marks = np.full(len(a), AtomMark.UNK)
    n_monomers = DefaultConfig["structure"]["n_monomers"]
    lig_pol_types = DefaultConfig["structure"]["ligand_pol_types"]
    # Keep track of polymer masks and sizes for each polymer type
    polymers = defaultdict(list)
    polymer_sizes = defaultdict(int)

    # Iterate over connected components (molecules) in a graph.
    for cc_idx in map(list, rx.connected_components(g)):
        # Make residue mask corresponding to all atoms from residues
        # of the connected component atom indices
        r_mask = util.extend_residue_mask(a, cc_idx)
        n_resi = bst.get_residue_count(a[r_mask])

        # Check if the single residue CC is a solvent.
        # Otherwise, it is a ligand candidate.
        if n_resi == 1:
            res_name = a[r_mask].res_name[0]
            if res_name in DefaultConfig["residues"]["solvents"]:
                marks[r_mask] = AtomMark.SOLVENT
            continue

        # If not solvent or polymer: continue
        if n_resi < n_monomers:
            continue

        pol_type = util.find_first_polymer_type(a[r_mask], n_monomers)
        if pol_type != "x":
            polymers[pol_type].append(r_mask)
            polymer_sizes[pol_type] += n_resi

    if not polymers or all(len(x) == 0 for x in polymers.values()):
        LOGGER.warning("No polymer atoms identified. Marking as SOLVENT+UNK.")
        return marks, "x", []

    # Determine primary polymer type
    cfg_pol_type = DefaultConfig["structure"]["primary_pol_type"][0]
    if cfg_pol_type == "a":
        prim_pol_type = max(polymer_sizes.items(), key=lambda x: x[1])[0]
    else:
        prim_pol_type = cfg_pol_type

    if not polymers[prim_pol_type]:
        LOGGER.warning(
            f"No primary polymer '{prim_pol_type}' atoms. Marking as SOLVENT+UNK."
        )
        return marks, "x", []

    # Force a single polymer chain or allow multiple ones
    fn = _to_single_poly_mask if single_poly_chain else _combine_poly_masks
    is_pol = fn(a, polymers[prim_pol_type])

    if not np.any(is_pol):
        LOGGER.warning(
            f"No primary polymer '{prim_pol_type}' atoms. Marking as SOLVENT+UNK."
        )
        return marks, "x", []

    # Mark determined primary polymer atoms. Ligand polymers will be marked below.
    marks[is_pol] = _POL_MARKS[prim_pol_type]

    ligands = []

    # Detect covalently bound solvent or ligand atoms.
    het_pol_idx = np.where(a.hetero & is_pol)[0]
    if len(het_pol_idx) > 0:
        starts_het_pol = np.unique(bst.get_residue_starts_for(a, het_pol_idx))
        for r_mask in bst.get_residue_masks(a, starts_het_pol):
            het_pol_type = util.find_first_polymer_type(a[r_mask], min_size=1)
            if het_pol_type == prim_pol_type:
                continue
            lig = make_ligand(r_mask, is_pol & ~r_mask, s)
            if lig is None:
                marks[r_mask] = AtomMark.SOLVENT | AtomMark.COVALENT
            else:
                marks[r_mask] = AtomMark.LIGAND | AtomMark.COVALENT
                ligands.append(lig)

    # Find unmarked atom indices and impose a subgraph based on them
    remaining_idx = np.where(marks == AtomMark.UNK)[0]
    sg = g.subgraph(remaining_idx)
    cc_idx_viewed = set()

    # Iterate over the subgraph's connected components
    for cc_idx in map(list, rx.connected_components(sg)):
        # Obtain original indices; they are reset upon a subgraph creation
        cc_idx = sg.subgraph(cc_idx).nodes()
        # Obtain a mask pointing to CC residues that were not previously
        # marked as polymer or solvent
        r_mask = util.extend_residue_mask(a, cc_idx)
        # Avoid creating duplicated ligands by storing already assessed atom
        # indices
        r_mask[list(cc_idx_viewed)] = False
        if not np.any(r_mask):
            continue
        cc_idx_viewed |= set(np.where(r_mask)[0])

        n_resi = bst.get_residue_count(a[r_mask])

        # Attempt making a ligand
        lig = make_ligand(r_mask, is_pol, s)
        if lig is None:
            continue

        multiples_flag = False  # indicates whether ligand is split below

        if n_resi > 1:  # Handle polymer ligand
            if lig.res_name[0] in lig_pol_types:
                marks[r_mask] = AtomMark.LIGAND | _POL_MARKS[lig.res_name[0]]
            else:
                # Ligand is non-polymer but still has multiple residues.
                # Users are warned about such cases.

                # Split into separate residues if polymer type is undetermined.
                # It's likely that a ligand appears as a single CC due to
                # structure artifacts, eg, molecules with different altloc
                # occupying the same spatial region.
                if lig.res_name[0] == "x":
                    multiples_flag = True
                    starts = np.unique(bst.get_residue_starts_for(a, cc_idx))
                    for r_mask in bst.get_residue_masks(a, starts):
                        lig = make_ligand(r_mask, is_pol & ~r_mask, s)
                        if lig is not None:
                            marks[r_mask] = AtomMark.LIGAND
                            ligands.append(lig)
                # Otherwise, despite polymeric nature of a ligand, it's
                # unsupported by the config and will be marked as a regular
                # ligand.
                else:
                    marks[r_mask] = AtomMark.LIGAND
        else:  # Handle a non-polymer ligand
            marks[r_mask] = AtomMark.LIGAND

        if not multiples_flag:
            ligands.append(lig)

    ligand_mask = reduce(
        op.or_,
        (lig.mask for lig in ligands),
        np.zeros_like(a.res_id, dtype=bool),
    )
    for lig in ligands:
        lig.contact_mask[ligand_mask] = False
        lig.dist[ligand_mask] = -1

    return marks, prim_pol_type, ligands


if __name__ == "__main__":
    raise RuntimeError
