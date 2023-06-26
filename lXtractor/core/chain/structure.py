from __future__ import annotations

import logging
import typing as t
from collections import abc
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path

import numpy as np
import pandas as pd
from biotite import structure as bst
from more_itertools import unzip
from typing_extensions import Self

from lXtractor.core import Ligand
from lXtractor.core.base import ApplyT, FilterT
from lXtractor.core.chain.base import topo_iter
from lXtractor.core.chain.list import _wrap_children, ChainList
from lXtractor.core.chain.sequence import ChainSequence
from lXtractor.core.config import (
    SeqNames,
    Sep,
    MetaNames,
    DumpNames,
    _DumpNames,
    EMPTY_PDB_ID,
    EMPTY_CHAIN_ID,
    UNK_NAME,
    ColNames,
)
from lXtractor.core.exceptions import LengthMismatch, InitError, MissingData
from lXtractor.core.structure import GenericStructure
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.structure import filter_selection, filter_solvent_extended

if t.TYPE_CHECKING:
    from lXtractor.variables import Variables

__all__ = ("ChainStructure", "filter_selection_extended", "subset_to_matching")

LOGGER = logging.getLogger(__name__)


# TODO: subset and overlap with other structures/sequences


@dataclass
class PDB_Chain:
    """
    A container to hold the data of a single structural chain.
    """

    id: str
    chain: str
    structure: GenericStructure | None


def _validate_chain(pdb: PDB_Chain):
    if pdb.structure.is_empty or pdb.structure.is_singleton:
        return
    chains = pdb.structure.chain_ids_polymer
    if len(chains) > 1:
        raise InitError(
            f"Invalid chain {pdb}: the structure must contain a single "
            f"polymeric chain. Got {len(chains)}: {chains}"
        )
    try:
        chain_id = (
            pdb.structure.chain_ids_polymer.pop()
            if pdb.structure.chain_ids_polymer
            else pdb.structure.chain_ids.pop()
        )
    except KeyError as e:
        raise InitError(f"No chains for {pdb}") from e
    if chain_id != pdb.chain:
        raise InitError(
            f"Invalid chain {pdb}. Actual chain {chain_id} does not match "
            f"chain attribute {pdb.chain}"
        )
    alt_loc = list(filter(lambda x: x != "", pdb.structure.altloc_ids))
    if len(alt_loc) > 1:
        raise InitError(
            f"Invalid chain {pdb}: the structure must contain a single alt loc; "
            f"found {len(alt_loc)} {alt_loc}"
        )


class ChainStructure:
    """
    A structure of a single chain.

    Typical usage workflow:

    #. Use :meth:`GenericStructure.read <lXtractor.core.structure.
        GenericStructure.read>` to parse the file.
    #. Split into chains using :meth:`split_chains <lXtractor.core.structure.
        GenericStructure.split_chains>`.
    #. Initialize :class:`ChainStructure` from each chain via
        :meth:`from_structure`.

    .. code-block:: python

        s = GenericStructure.read(Path("path/to/structure.cif"))
        chain_structures = [
            ChainStructure.from_structure(c) for c in s.split_chains()
        ]

    Two main containers are:

    1) :attr:`seq` -- a :class:`ChainSequence` of this structure,
        also containing meta info.
    2) :attr:`pdb` -- a container with pdb id, pdb chain id,
        and the structure itself.

    A unique structure is defined by
    """

    __slots__ = ("pdb", "seq", "parent", "variables", "children")

    def __init__(
        self,
        pdb_id: str,
        pdb_chain: str,
        pdb_structure: GenericStructure | None,
        seq: ChainSequence | None = None,
        parent: ChainStructure | None = None,
        children: abc.Iterable[ChainStructure] | None = None,
        variables: Variables | None = None,
        skip_validation: bool = False,
    ):
        """
        `pdb_id`, `pdb_chain`, and `pdb_structure` are wrapped into a
        :attr:`pdb`: -- a :class:`PDB_Chain` container.

        :param pdb_id: Four-letter PDB code.
        :param pdb_chain: PDB Chain code.
        :param pdb_structure: Parsed generic structure with a single chain.
        :param seq: Chain sequence of a structure. If not provided, will use
            :meth:`get_sequence <lXtractor.core.structure.
            GenericStructure.get_sequence>`.
        :param parent: Specify parental structure.
        :param children: Specify structures descended from this one.
            This contained is used to record sub-structures obtained via
            :meth:`spawn_child`.
        :param variables: Variables associated with this structure.
        :param skip_validation: Skip validating that only a single chain is
            present. This might be useful when there is a need to initialize
            chain structure from a peculiar subset of atoms.
        :raise InitError: If invalid (e.g., multi-chain structure) is provided.
        """
        from lXtractor.variables import Variables

        #: A container with PDB ID, PDB Chain, and parsed structure.
        if pdb_structure is None:
            pdb_structure = GenericStructure.make_empty(pdb_id)
        self.pdb: PDB_Chain = PDB_Chain(pdb_id, pdb_chain, pdb_structure)
        if not skip_validation:
            _validate_chain(self.pdb)

        #: Sequence of this structure.
        self.seq: ChainSequence

        #: Parent of this structure.
        self.parent: ChainStructure | None = parent

        #: Variables assigned to this structure. Each should be of a
        #: :class:`lXtractor.variables.base.StructureVariable`.
        self.variables: Variables = variables or Variables()

        #: Any sub-structures descended from this one,
        #: preferably using :meth:`spawn_child`.
        self.children: ChainList[ChainStructure] = _wrap_children(children)

        if seq is None:
            if len(self.pdb.structure.array) > 0:
                str_seq = list(self.pdb.structure.get_sequence())
            else:
                str_seq = []
            seq1: list[str] = [x[0] for x in str_seq]
            seq3: list[str] = [x[1] for x in str_seq]
            num: list[int] = [x[2] for x in str_seq]
            seqs: dict[str, list[int] | list[str]] = {
                SeqNames.seq3: seq3,
                SeqNames.enum: num,
            }
            self.seq = ChainSequence.from_string(
                "".join(seq1),
                name=f"{pdb_id}{Sep.chain}{pdb_chain}",
                **seqs,  # type: ignore  # Fails to recognize kwargs
            )
        else:
            self.seq = seq

        self.seq.meta[MetaNames.pdb_id] = pdb_id
        self.seq.meta[MetaNames.pdb_chain] = pdb_chain
        self.seq.meta[MetaNames.altloc] = self.altloc

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def __len__(self) -> int:
        if self.pdb.structure is None:
            return 0
        return len(self.array)

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, ChainStructure):
            return self.pdb == other.pdb and self.seq == other.seq
        return False

    def __hash__(self) -> int:
        return (
            hash(self.seq)
            + hash(self.pdb.id)
            + hash(self.pdb.chain)
            + hash(self.pdb.structure)
        )

    @property
    def id(self) -> str:
        """
        :return: ChainStructure identifier in the format
            "ChainStructure({seq.id}|{alt_locs})<-(parent.id)".
        """
        alt_locs = self.pdb.structure.id.split("|")[-1]
        parent = "" if self.parent is None else f"<-({self.parent.id})"
        return f"ChainStructure({self.seq.id}|{alt_locs}){parent}"

    @property
    def array(self) -> bst.AtomArray:
        """
        :return: The ``AtomArray`` object (a shortcut for
            ``.pdb.structure.array``).
        """
        return self.pdb.structure.array

    @property
    def meta(self) -> dict[str, str]:
        """
        :return: Meta info of a :attr:`seq`.
        """
        return self.seq.meta

    @property
    def start(self) -> int:
        """
        :return: Structure sequence's :attr:`start <lXtractor.core.chain.
            sequence.start>`
        """
        return self.seq.start

    @property
    def end(self) -> int:
        """
        :return: Structure sequence's :attr:`end <lXtractor.core.chain.
            sequence.end>`
        """
        return self.seq.end

    @property
    def name(self) -> str | None:
        """
        :return: Structure sequence's :attr:`name <lXtractor.core.chain.
            sequence.name>`
        """
        return self.seq.name

    @property
    def categories(self) -> list[str]:
        """
        :return: A list of categories encapsulated within
            :attr:`ChainSequence.meta`.
        """
        return self.seq.categories

    @property
    def is_empty(self) -> bool:
        """
        :return: ``True`` if the structure is empty and ``False`` otherwise.
        """
        return len(self) == 0

    @property
    def ligands(self) -> list[Ligand]:
        """
        :return: A list of connected ligands.
        """
        return self.pdb.structure.ligands

    @property
    def altloc(self) -> str:
        """
        :return: An altloc ID.
        """
        return self.pdb.structure.altloc_ids[0]

    @classmethod
    def from_structure(
        cls,
        structure: bst.AtomArray | GenericStructure,
        pdb_id: str | None = None,
        chain_id: str | None = None,
        **kwargs,
    ) -> ChainStructure:
        """
        :param structure: An `AtomArray` or `GenericStructure`,
            corresponding to a single protein chain.
        :param pdb_id: PDB identifier of a structure
            (Chain ID will be inferred from the `AtomArray`).
        :param chain_id: Chain identifier. If not provided, will take the first
            atom's chain ID from the provided structure.
        :param kwargs: Passed to initializer. Avoid using `pdb_id`, `chain_id`,
            or `structure` since they are supplied internally within this
            method.
        :return: Initialized chain structure.
        """
        if isinstance(structure, bst.AtomArray):
            structure = GenericStructure(structure, pdb_id)

        pdb_id = pdb_id or structure.pdb_id or EMPTY_PDB_ID

        if structure.is_empty:
            return cls(pdb_id, chain_id or EMPTY_CHAIN_ID, structure)

        chain_id = chain_id or (
            structure.chain_ids_polymer.pop()
            if structure.chain_ids_polymer
            else structure.chain_ids.pop()
        )

        return cls(pdb_id, chain_id, structure, **kwargs)

    @classmethod
    def make_empty(
        cls, pdb_id: str = EMPTY_PDB_ID, pdb_chain: str = EMPTY_CHAIN_ID, **kwargs
    ) -> ChainStructure:
        """
        Create an empty chain structure.

        :param pdb_id: PDB ID.
        :param pdb_chain: Chain ID.
        :param kwargs: Passed to initializer. Avoid using `pdb_id`, `chain_id`,
            or `structure` since they are supplied internally within this
            method.
        :return: An empty chain structure.
        """
        return cls(pdb_id, pdb_chain, GenericStructure.make_empty(pdb_id), **kwargs)

    def rm_solvent(self) -> Self:
        """
        Remove solvent "residues" from this structure.

        :return: A new instance without solvent molecules.
        """
        if self.is_empty:
            return self

        return self.__class__.from_structure(
            GenericStructure(
                self.array[~filter_solvent_extended(self.array)], self.pdb.id
            ),
            self.pdb.id,
            self.pdb.chain,
        )

    def superpose(
        self,
        other: ChainStructure,
        res_id: abc.Sequence[int] | None = None,
        atom_names: abc.Sequence[abc.Sequence[str]] | abc.Sequence[str] | None = None,
        map_name_self: str | None = None,
        map_name_other: str | None = None,
        mask_self: np.ndarray | None = None,
        mask_other: np.ndarray | None = None,
        inplace: bool = False,
        rmsd_to_meta: bool = True,
    ) -> tuple[ChainStructure, float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Superpose some other structure to this one.
        It uses func:`biotite.structure.superimpose` internally.

        The most important requirement is both structures (after all optional
        selections applied) having the same number of atoms.

        :param other: Other chain structure (mobile).
        :param res_id: Residue positions within this or other chain structure.
            If ``None``, use all available residues.
        :param atom_names: Atom names to use for selected residues. Two options
            are available:

            1) Sequence of sequences of atom names. In this case, atom names
            are given per selected residue (`res_id`), and the external
            sequence's length must correspond to the number of residues in the
            `res_id`. Note that if no `res_id` provided, the sequence must
            encompass all available residues.

            2) A sequence of atom names. In this case, it will be used
            to select atoms for each available residues. For instance, use
            ``atom_names=["CA", "C", "N"]`` to select backbone atoms.

        :param map_name_self: Use this map to map `res_id` to real numbering
            of this structure.
        :param map_name_other: Use this map to map `res_id` to real numbering
            of the `other` structure.
        :param mask_self: Per-atom boolean selection mask to pick fixed atoms
            within this structure.
        :param mask_other: Per-atom boolean selection mask to pick mobile atoms
            within the `other` structure. Note that `mask_self` and
            `mask_other` take precedence over other selection specifications.
        :param inplace: Apply the transformation to the mobile structure
            inplace, mutating `other`. Otherwise, make a new instance:
            same as `other`, but with transformed atomic coordinates of
            a :attr:`pdb.structure`.
        :param rmsd_to_meta: Write RMSD to the :attr:`meta` of `other` as "rmsd
        :return: A tuple with (1) transformed chain structure,
            (2) transformation RMSD, and (3) transformation matrices
            (see func:`biotite.structure.superimpose` for details).
        """

        def _get_mask(
            c: ChainStructure,
            map_name: str | None,
            _atom_names: abc.Sequence[abc.Sequence[str]] | None,
        ) -> np.ndarray:
            if res_id is None:
                return np.ones_like(c.array, bool)

            if not map_name or res_id is None:
                _res_id = res_id
            else:
                mapping = c.seq.get_map(map_name)
                _res_id = [mapping[x]._asdict()[SeqNames.enum] for x in res_id]

            return filter_selection(c.array, _res_id, _atom_names)

        if self.is_empty or other.is_empty:
            raise MissingData("Overlapping empty structures is not supported")

        match atom_names:
            case [str(), *_]:
                if res_id is not None:
                    # Fails to infer Sequence[str] type
                    atom_names = [atom_names] * len(res_id)  # type: ignore
            case [[str(), *_], *_]:
                if res_id is not None and len(res_id) != len(atom_names):
                    raise LengthMismatch(
                        "When specifying `atom_names` per residue, the number of "
                        f"residues must match the number of atom name groups; "
                        f"Got {len(res_id)} residues and {len(atom_names)} "
                        "atom names groups."
                    )

        if mask_self is None:
            mask_self = _get_mask(self, map_name_self, atom_names)
        if mask_other is None:
            mask_other = _get_mask(other, map_name_other, atom_names)

        superposed, rmsd, transformation = self.pdb.structure.superpose(
            other.pdb.structure, mask_self=mask_self, mask_other=mask_other
        )

        if inplace:
            other.pdb.structure = superposed
        else:
            other = ChainStructure(
                other.pdb.id,
                other.pdb.chain,
                superposed,
                other.seq,
                parent=other.parent,
                children=other.children,
                variables=other.variables,
            )

        if rmsd_to_meta:
            # TODO: Is this correct?
            map_name = map_name_other or other.seq.name
            other.seq.meta[f"rmsd_{map_name}"] = rmsd

        return other, rmsd, transformation

    def spawn_child(
        self,
        start: int,
        end: int,
        name: str | None = None,
        *,
        map_from: str | None = None,
        map_closest: bool = True,
        keep_seq_child: bool = False,
        keep: bool = True,
        deep_copy: bool = False,
    ) -> ChainStructure:
        """
        Create a sub-structure from this one.
        `Start` and `end` have inclusive boundaries.

        :param start: Start coordinate.
        :param end: End coordinate.
        :param name: The name of the spawned sub-structure.
        :param map_from: Optionally, the map name the boundaries correspond to.
        :param map_closest: Map to closest `start`, `end` boundaries
            (see :meth:`map_boundaries`).
        :param keep_seq_child: Keep spawned sub-sequence within
            :attr:`ChainSequence.children`. Beware that it's best to use
            a single object type for keeping parent-children relationships
            to avoid duplicating information.
        :param keep: Keep spawned substructure in :attr:`children`.
        :param deep_copy: Deep copy spawned sub-sequence and sub-structure.
        :return: New chain structure -- a sub-structure of the current one.
        """

        if start > end:
            raise ValueError(f"Invalid boundaries {start, end}")

        if self.is_empty:
            raise MissingData("Attempting to spawn a child from an empty structure")

        name = name or self.seq.name

        seq = self.seq.spawn_child(
            start,
            end,
            name,
            map_from=map_from,
            map_closest=map_closest,
            deep_copy=deep_copy,
            keep=keep_seq_child,
        )

        enum_field = seq.field_names().enum
        start, end = seq[enum_field][0], seq[enum_field][-1]
        structure = self.pdb.structure.extract_segment(start, end)

        child = ChainStructure(self.pdb.id, self.pdb.chain, structure, seq, self)
        if keep:
            self.children.append(child)
        return child

    def iter_children(self) -> abc.Generator[list[ChainStructure], None, None]:
        """
        Iterate :attr:`children` in topological order.

        See :meth:`ChainSequence.iter_children` and :func:`topo_iter`.
        """
        return topo_iter(self, lambda x: x.children)

    def apply_children(self, fn: ApplyT[ChainStructure], inplace: bool = False) -> Self:
        """
        Apply some function to children.

        :param fn: A callable accepting and returning the chain structure type
            instance.
        :param inplace: Apply to children in place. Otherwise, return a copy
            with only children transformed.
        :return: A chain structure with transformed children.
        """
        children = self.children.apply(fn)
        if inplace:
            self.children = children
            return self
        return self.__class__(
            self.pdb.id,
            self.pdb.chain,
            self.pdb.structure,
            seq=self.seq,
            children=children,
            parent=self.parent,
            variables=self.variables,
        )

    def filter_children(
        self, pred: FilterT[ChainStructure], inplace: bool = False
    ) -> Self:
        """
        Filter children using some predicate.

        :param pred: Some callable accepting chain structure and returning
            bool.
        :param inplace: Filter :attr:`children` in place. Otherwise, return
            a copy with only children transformed.
        :return: A chain structure with filtered children.
        """
        children = self.children.filter(pred)
        if inplace:
            self.children = children
            return self
        return self.__class__(
            self.pdb.id,
            self.pdb.chain,
            self.pdb.structure,
            seq=self.seq,
            children=children,
            parent=self.parent,
            variables=self.variables,
        )

    @classmethod
    def read(
        cls,
        base_dir: Path,
        *,
        dump_names: _DumpNames = DumpNames,
        search_children: bool = False,
        **kwargs,
    ) -> Self:
        """
        Read the chain structure from a file disk dump.

        :param base_dir: An existing dir containing structure,
            structure sequence, meta info, and (optionally) any sub-structure
            segments.
        :param dump_names: File names container.
        :param search_children: Recursively search for sub-segments and
            populate :attr:`children`.
        :param kwargs: Passed to
            :meth:`lXtractor.core.structure.GenericStructure.read`.
        :return: An initialized chain structure.
        """

        files = get_files(base_dir)
        dirs = get_dirs(base_dir)
        variables = None

        bname = dump_names.structure_base_name
        stems = {p.name.split(".")[0]: p.name for p in files.values()}
        if bname not in stems:
            raise InitError(
                f"{base_dir} must contain {bname}.fmt "
                f'where "fmt" is supported structure format'
            )

        seq = ChainSequence.read(base_dir, dump_names=dump_names, search_children=False)
        pdb_id = seq.meta.get(MetaNames.pdb_id, EMPTY_PDB_ID)
        chain_id = seq.meta.get(MetaNames.pdb_chain, EMPTY_CHAIN_ID)

        structure = GenericStructure.read(
            base_dir / stems[bname], structure_id=pdb_id, **kwargs
        )

        if structure.pdb_id == EMPTY_PDB_ID:
            structure.pdb_id = pdb_id

        if dump_names.variables in files:
            from lXtractor.variables import Variables

            variables = Variables.read(files[dump_names.variables]).structure

        cs = cls(pdb_id, chain_id, structure, seq, variables=variables)

        if search_children and dump_names.segments_dir in dirs:
            for path in (base_dir / dump_names.segments_dir).iterdir():
                child = cls.read(path, dump_names=dump_names, search_children=True)
                child.parent = cs
                cs.children.append(child)

        return cs

    def write(
        self,
        base_dir: Path,
        fmt: str = "mmtf.gz",
        *,
        dump_names: _DumpNames = DumpNames,
        write_children: bool = False,
    ):
        """
        Write this object into a directory. It will create the following files:

        #. meta.tsv
        #. sequence.tsv
        #. structure.fmt

        Existing files will be overwritten.

        :param base_dir: A writable dir to save files to.
        :param fmt: Structure format to use. Supported formats are "pdb", "cif",
            and "mmtf". Adding ".gz" (eg, "mmtf.gz") will lead to gzip
            compression.
        :param dump_names: File names container.
        :param write_children: Recursively write :attr:`children`.
        :return: Nothing.
        """

        if self.is_empty:
            raise MissingData("Attempting to write an empty chain structure")

        base_dir.mkdir(exist_ok=True, parents=True)

        self.seq.write(base_dir)
        self.pdb.structure.write(base_dir / f"{dump_names.structure_base_name}.{fmt}")
        if self.variables:
            self.variables.write(base_dir / dump_names.variables)

        if write_children:
            for child in self.children:
                child_name = child.seq.name or UNK_NAME
                child_dir = base_dir / DumpNames.segments_dir / child_name
                child.write(child_dir, fmt, dump_names=dump_names, write_children=True)

    def summary(
        self, meta: bool = True, children: bool = False, ligands: bool = False
    ) -> pd.DataFrame:
        s = self.seq.summary(meta=meta, children=False)
        s[ColNames.id] = [self.id]
        if ligands and len(self.ligands) > 0:
            lig_df = pd.DataFrame(lig.summary() for lig in self.ligands)
            lig_df.columns = ["Ligand_" + c for c in lig_df.columns]
            s = pd.concat([s, lig_df])
        if children and self.children:
            child_summaries = pd.concat(
                [c.summary(meta=meta, children=children) for c in self.children]
            )
            s = pd.concat([s, child_summaries])
        return s


def filter_selection_extended(
    c: ChainStructure,
    pos: abc.Sequence[int] | None = None,
    atom_names: abc.Sequence[abc.Sequence[str]] | abc.Sequence[str] | None = None,
    map_name: str | None = None,
    exclude_hydrogen: bool = False,
    tolerate_missing: bool = False,
) -> np.ndarray:
    """
    Get mask for certain positions and atoms of a chain structure.

    .. seealso:
        :func:`lXtractor.util.seq.filter_selection`

    :param c: Arbitrary chain structure.
    :param pos: A sequence of positions.
    :param atom_names: A sequence of atom names (broadcasted to each position
        in `res_id`) or an iterable over such sequences for each position
        in `res_id`.
    :param map_name: A map name to map from `pos` to
        :meth:`numbering <lXtractor.core.Chain.ChainSequence.numbering>`
    :param exclude_hydrogen: For convenience, exclude hydrogen atoms.
        Especially useful during pre-processing for superposition.
    :param tolerate_missing: If certain positions failed to map, does not
        raise an error.
    :return: A binary mask, ``True`` for selected atoms.
    """
    if pos is not None and map_name:
        _map = c.seq.get_map(map_name)

        mapped_pairs = [(p, _map.get(p, None)) for p in pos]

        if not tolerate_missing:
            for p, p_mapped in mapped_pairs:
                if p_mapped is None:
                    raise MissingData(f"Position {p} failed to map for {c}")

        pos = [x[1].numbering for x in mapped_pairs if x[1] is not None]

        if len(pos) == 0:
            LOGGER.warning("No positions were selected.")
            return np.zeros_like(c.array, dtype=bool)

    m = filter_selection(c.array, pos, atom_names)

    if exclude_hydrogen:
        m &= c.array.element != "H"

    return m


def subset_to_matching(
    reference: ChainStructure,
    c: ChainStructure,
    map_name: str | None = None,
    skip_if_match: str = SeqNames.seq1,
    **kwargs,
) -> tuple[ChainStructure, ChainStructure]:
    """
    Subset both chain structures to aligned residues using
    **sequence alignment**.

    .. note::
        It's not necessary, but it makes sense for `c1` and `c2` to be
        somehow related.

    :param reference: A chain structure to align to.
    :param c: A chain structure to align.
    :param map_name: If provided, `c` is considered "pre-aligned" to the
        `reference`, and `reference` possessed the numbering under `map_name`.
    :param skip_if_match: Two options:

        1. Sequence/Map name, e.g., "seq1" -- if sequences under this name
        match exactly, skip alignment and return original chain structures.

        2. "len" -- if sequences have equal length, skip alignment and return
        original chain structures.
    :return: A pair of new structures having the same number of residues
        that were successfully matched during the alignment.
    """
    if skip_if_match == "len":
        if len(reference.seq) == len(c.seq):
            return reference, c
    else:
        if reference.seq[skip_if_match] == c.seq[skip_if_match]:
            return reference, c

    pos_pairs: abc.Iterable[tuple[int, int | None]]
    if not map_name:
        pos2 = reference.seq.map_numbering(c.seq, **kwargs)
        pos1 = reference.seq[SeqNames.enum]
        pos_pairs = zip(pos1, pos2, strict=True)
    else:
        pos_pairs = zip(
            reference.seq[SeqNames.enum], reference.seq[map_name], strict=True
        )

    pos_pairs = filter(lambda x: x[0] is not None and x[1] is not None, pos_pairs)
    _pos1, _pos2 = unzip(pos_pairs)
    _pos1, _pos2 = map(list, [_pos1, _pos2])

    c1_new, c2_new = starmap(
        lambda s, pos: ChainStructure.from_structure(
            s.pdb.structure.extract_positions(pos, s.pdb.chain)
        ),
        [(reference, _pos1), (c, _pos2)],
    )

    return c1_new, c2_new


if __name__ == "__main__":
    raise RuntimeError
