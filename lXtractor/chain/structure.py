from __future__ import annotations

import logging
import typing as t
from collections import abc
from pathlib import Path

import numpy as np
import pandas as pd
from biotite import structure as bst
from more_itertools import unzip

from lXtractor.chain import ChainList, ChainSequence
from lXtractor.chain.base import topo_iter
from lXtractor.chain.list import _wrap_children
from lXtractor.core import Ligand
from lXtractor.core.base import ApplyT, FilterT
from lXtractor.core.config import DefaultConfig
from lXtractor.core.exceptions import LengthMismatch, InitError, MissingData
from lXtractor.core.structure import GenericStructure
from lXtractor.util import biotite_align
from lXtractor.util.io import get_files, get_dirs
from lXtractor.util.structure import filter_selection

if t.TYPE_CHECKING:
    from lXtractor.variables import Variables

__all__ = ("ChainStructure", "filter_selection_extended", "subset_to_matching")

LOGGER = logging.getLogger(__name__)


# TODO: subset and overlap with other structures/sequences


def _validate_chain(structure: GenericStructure):
    if structure.is_empty or structure.is_singleton:
        return
    chains = structure.chain_ids_polymer
    if len(chains) > 1:
        raise InitError(
            f"The structure {structure} must contain a single "
            f"protein chain. Got {len(chains)}: {chains}"
        )
    # try:
    #     chain_id = (
    #         pdb.structure.chain_ids_polymer.pop()
    #         if pdb.structure.chain_ids_polymer
    #         else pdb.structure.chain_ids.pop()
    #     )
    # except KeyError as e:
    #     raise InitError(f"No chains for {pdb}") from e
    # if chain_id != pdb.chain:
    #     raise InitError(
    #         f"Invalid chain {pdb}. Actual chain {chain_id} does not match "
    #         f"chain attribute {pdb.chain}"
    #     )
    alt_loc = list(filter(lambda x: x != "", structure.altloc_ids))
    if len(alt_loc) > 1:
        raise InitError(
            f"The structure {structure} must contain a single alt loc; "
            f"found {len(alt_loc)} {alt_loc}"
        )


def _validate_chain_seq(
    structure: GenericStructure, seq: ChainSequence, report_aln: bool = True
) -> None:
    str_seq = _str2seq(structure, None, None)
    if not seq.seq1 == str_seq.seq1:
        msg = (
            f"Primary sequences of structure's {structure} sequence "
            f"and sequence {seq} mismatch."
        )
        if report_aln:
            (_, s1), (_, s2) = biotite_align(
                [(structure.id, str_seq.seq1), (seq.id, seq.seq1)]
            )
            msg += f"\n>{structure.id}\n{s1}\n{seq.id}\n{s2}"
        raise InitError(msg)


def _get_chain_id(structure: GenericStructure):
    if structure.is_empty:
        return DefaultConfig["unknowns"]["chain_id"]
    chain_ids = structure.chain_ids_polymer or structure.chain_ids
    if len(chain_ids) == 0:
        raise MissingData(f"Cannot determine chain ID of structure {structure}")
    return chain_ids.pop()


def _str2seq(
    structure: GenericStructure, str_id: str | None, chain_id: str | None
) -> ChainSequence:
    chain_id = chain_id or _get_chain_id(structure)
    str_id = str_id or structure.name
    sep_chain = DefaultConfig["separators"]["chain"]
    name = f"{str_id}{sep_chain}{chain_id}"

    str_seq = list(structure.get_sequence())
    if not str_seq:
        return ChainSequence.make_empty(name=name)

    seq1, seq3, num = map(list, unzip(str_seq))
    seqs: dict[str, list[int] | list[str]] = {
        DefaultConfig["mapnames"]["seq3"]: seq3,
        DefaultConfig["mapnames"]["enum"]: num,
    }

    return ChainSequence.from_string("".join(seq1), name=name, **seqs)


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

    1) :attr:`_seq` -- a :class:`ChainSequence` of this structure,
        also containing meta info.
    2) :attr:`pdb` -- a container with pdb id, pdb chain id,
        and the structure itself.

    A unique structure is defined by
    """

    __slots__ = (
        "_id",
        "_structure",
        "_chain_id",
        "_seq",
        "_parent",
        "variables",
        "children",
    )

    def __init__(
        self,
        structure: GenericStructure | bst.AtomArray | None,
        chain_id: str | None = None,
        structure_id: str | None = None,
        seq: ChainSequence | None = None,
        parent: ChainStructure | None = None,
        children: abc.Iterable[ChainStructure] | None = None,
        variables: Variables | None = None,
    ):
        """

        :param structure_id: An ID for the structure the chain was taken from.
        :param chain_id: A chain ID (e.g., "A", "B", etc.)
        :param structure: Parsed generic structure with a single chain.
        :param seq: Chain sequence of a structure. If not provided, will use
            :meth:`get_sequence <lXtractor.core.structure.
            GenericStructure.get_sequence>`.
        :param parent: Specify parental structure.
        :param children: Specify structures descended from this one.
            This contained is used to record sub-structures obtained via
            :meth:`spawn_child`.
        :param variables: Variables associated with this structure.
        :raise InitError: If invalid (e.g., multi-chain structure) is provided.
        """
        from lXtractor.variables import Variables

        if isinstance(structure, bst.AtomArray):
            structure = GenericStructure(
                structure, structure_id or DefaultConfig["unknowns"]["structure_id"]
            )

        if structure is None:
            structure = GenericStructure.make_empty(structure_id)
            self._chain_id = chain_id or DefaultConfig["unknowns"]["chain_id"]
        else:
            _validate_chain(structure)
            self._chain_id = _get_chain_id(structure)

        self._structure = structure

        structure_id = structure_id or structure.name

        #: Variables assigned to this structure. Each should be of a
        #: :class:`lXtractor.variables.base.StructureVariable`.
        self.variables: Variables = variables or Variables()

        #: Any sub-structures descended from this one,
        #: preferably using :meth:`spawn_child`.
        self.children: ChainList[ChainStructure] = _wrap_children(children)

        if seq is None:
            self._seq = _str2seq(structure, structure_id, self.chain_id)
        else:
            if not structure.is_empty:
                _validate_chain_seq(structure, seq)
            self._seq = seq

        self._parent: ChainStructure | None = parent

        self._id = self._make_id()

        names = DefaultConfig["metadata"]
        self.seq.meta[names["structure_id"]] = structure_id
        self.seq.meta[names["structure_chain_id"]] = chain_id
        self.seq.meta[names["altloc"]] = self.altloc

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    def __len__(self) -> int:
        if self.structure is None or self.structure.is_empty:
            return 0
        return len(self.seq)

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, ChainStructure):
            return (
                self.id == other.id
                and self.seq == other.seq
                and self.structure == other.structure
            )
        return False

    def __hash__(self) -> int:
        return hash(self.id)
        # return hash(self.seq) + hash(self.structure)

    @property
    def chain_id(self) -> str:
        return self._chain_id

    @chain_id.setter
    def chain_id(self, value: str):
        if not isinstance(value, str):
            raise TypeError(f"Chain ID must be of type str; got {type(value)}")
        self._chain_id = value
        self._id = self._make_id()

    @property
    def structure(self) -> GenericStructure:
        return self._structure

    @structure.setter
    def structure(self, value: GenericStructure):
        if not isinstance(value, GenericStructure):
            raise TypeError(
                f"Invalid type {type(value)} to set the structure attribute"
            )
        _validate_chain_seq(value, self.seq)
        self._structure = value
        self._chain_id = _get_chain_id(value)
        self._id = self._make_id()

    @property
    def seq(self) -> ChainSequence:
        return self._seq

    @seq.setter
    def seq(self, value: ChainSequence):
        if not isinstance(value, ChainSequence):
            raise TypeError(f"Invalid value type {type(value)}")
        _validate_chain_seq(self.structure, value)
        self._seq = value
        self._id = self._make_id()

    @property
    def parent(self) -> t.Self | None:
        return self._parent

    @parent.setter
    def parent(self, value: t.Self | None):
        if not isinstance(value, (type(self), type(None))):
            raise TypeError(
                f"Parent must be of the same type {type(self)}. " f"Got {type(value)}"
            )
        self._parent = value
        self._id = self._make_id()

    def _make_id(self) -> str:
        alt_locs = self.structure.id.split("|")[-1]
        parent = "" if self.parent is None else f"<-({self.parent.id})"
        return f"ChainStructure({self.seq.id_strip_parents()}|{alt_locs}){parent}"

    @property
    def id(self) -> str:
        """
        :return: ChainStructure identifier in the format
            "ChainStructure({_seq.id}|{alt_locs})<-(parent.id)".
        """
        return self._id

    @property
    def array(self) -> bst.AtomArray:
        """
        :return: The ``AtomArray`` object (a shortcut for
            ``.pdb.structure.array``).
        """
        return self.structure.array

    @property
    def meta(self) -> dict[str, str]:
        """
        :return: Meta info of a :attr:`_seq`.
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
    def ligands(self) -> tuple[Ligand, ...]:
        """
        :return: A list of connected ligands.
        """
        return self.structure.ligands

    @property
    def altloc(self) -> str:
        """
        :return: An altloc ID.
        """
        return self.structure.altloc_ids[0]

    @classmethod
    def make_empty(cls) -> ChainStructure:
        """
        Create an empty chain structure.

        :return: An empty chain structure.
        """
        return cls(None)

    def rm_solvent(self, copy: bool = False) -> t.Self:
        """
        Remove solvent "residues" from this structure.

        :param copy: Copy an atom array that results from solvent removal.
        :return: A new instance without solvent molecules.
        """
        if self.is_empty:
            return self

        return self.__class__(
            self.structure.rm_solvent(copy=copy),
            self.chain_id,
            seq=self.seq,
            parent=self.parent,
            children=self.children,
            variables=self.variables,
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
                _res_id = [
                    mapping[x]._asdict()[DefaultConfig["mapnames"]["enum"]]
                    for x in res_id
                ]

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

        superposed, rmsd, transformation = self.structure.superpose(
            other.structure, mask_self=mask_self, mask_other=mask_other
        )

        if inplace:
            other.structure = superposed
        else:
            other = ChainStructure(
                other.structure,
                other.chain_id,
                other.structure.name,
                other.seq,
                other.parent,
                other.children,
                other.variables,
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
        category: str | None = None,
        *,
        map_from: str | None = None,
        map_closest: bool = True,
        keep_seq_child: bool = False,
        keep: bool = True,
        deep_copy: bool = False,
        tolerate_failure: bool = False,
        silent: bool = False,
    ) -> ChainStructure:
        """
        Create a sub-structure from this one.
        `Start` and `end` have inclusive boundaries.

        :param start: Start coordinate.
        :param end: End coordinate.
        :param name: The name of the spawned sub-structure.
        :param category: Spawned child category. Any meaningful tag string that
            could be used later to group similar children.
        :param map_from: Optionally, the map name the boundaries correspond to.
        :param map_closest: Map to closest `start`, `end` boundaries
            (see :meth:`map_boundaries`).
        :param keep_seq_child: Keep spawned sub-sequence within
            :attr:`ChainSequence.children`. Beware that it's best to use
            a single object type for keeping parent-children relationships
            to avoid duplicating information.
        :param keep: Keep spawned substructure in :attr:`children`.
        :param deep_copy: Deep copy spawned sub-sequence and sub-structure.
        :param tolerate_failure: Do not raise the ``InitError` if the resulting
            structure subset is empty,
        :param silent: Do not display warnings if `tolerate_failure` is ``True``.
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
            category,
            map_from=map_from,
            map_closest=map_closest,
            deep_copy=deep_copy,
            keep=keep_seq_child,
        )

        enum_field = DefaultConfig["mapnames"]["enum"]
        start, end = seq[enum_field][0], seq[enum_field][-1]
        structure = self.structure.extract_segment(start, end, self.chain_id)

        if (
            structure.is_empty
            or structure.is_empty_polymer
            and not structure.is_singleton
        ):
            if tolerate_failure:
                if not silent:
                    LOGGER.warning(
                        f"Extracting structure segment using boundaries "
                        f"({start}, {end}) yielded an empty structure."
                    )
            else:
                raise InitError("The resulting substructure is empty")
        else:
            # In some cases the extracted structure sequence is slightly different
            # from the spawned segment (e.g., when the segment's end is a single
            # disjoint residue and therefore not treated as a part of a polymer).
            # To be safe, we adjust spawned segment boundaries to avoid mismatched
            # primary sequences in polymeric peptide and extracted segment.
            a = structure.array[structure.mask.primary_polymer]
            if len(a) == 0:
                a = structure.array
            enum2segment = seq.get_map(enum_field, "i")
            try:
                seq_start = enum2segment[a.res_id[0]]
                seq_end = enum2segment[a.res_id[-1]]
            except (KeyError, IndexError) as e:
                raise MissingData(
                    f"Failed to adjust extracted sequence boundaries for spawned "
                    f"child sequence {seq}."
                ) from e
            seq = seq[seq_start:seq_end]

        child = ChainStructure(
            structure, self.chain_id, self.structure.name, seq=seq, parent=self
        )
        if keep:
            self.children.append(child)
        return child

    def iter_children(self) -> abc.Generator[list[ChainStructure], None, None]:
        """
        Iterate :attr:`children` in topological order.

        See :meth:`ChainSequence.iter_children` and :func:`topo_iter`.
        """
        return topo_iter(self, lambda x: x.children)

    def apply_children(
        self, fn: ApplyT[ChainStructure], inplace: bool = False
    ) -> t.Self:
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
            self.structure,
            self.chain_id,
            seq=self.seq,
            children=children,
            parent=self.parent,
            variables=self.variables,
        )

    def filter_children(
        self, pred: FilterT[ChainStructure], inplace: bool = False
    ) -> t.Self:
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
            self.structure,
            self.chain_id,
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
        search_children: bool = False,
        **kwargs,
    ) -> t.Self:
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

        fnames = DefaultConfig["filenames"]
        mnames = DefaultConfig["metadata"]
        unk = DefaultConfig["unknowns"]
        bname = fnames["structure_base_name"]
        stems = {
            p.name.split(".")[0]: p.name
            for p in files.values()
            if p.suffix not in [".npy", ".json"]
        }
        if bname not in stems:
            raise InitError(
                f"{base_dir} must contain {bname}.fmt "
                f'where "fmt" is supported structure format'
            )
        seq = ChainSequence.read(base_dir, search_children=False)
        s_id = seq.meta.get(mnames["structure_id"], unk["structure_id"])
        chain_id = seq.meta.get(mnames["structure_chain_id"], unk["chain_id"])

        if "structure_id" not in kwargs:
            kwargs["structure_id"] = s_id

        structure = GenericStructure.read(base_dir / stems[bname], **kwargs)

        if fnames["variables"] in files:
            from lXtractor.variables import Variables

            variables = Variables.read(files[fnames["variables"]]).structure

        cs = cls(structure, chain_id, seq=seq, variables=variables)

        if search_children and fnames["segments_dir"] in dirs:
            for path in (base_dir / fnames["segments_dir"]).iterdir():
                child = cls.read(path, search_children=True)
                child.parent = cs
                cs.children.append(child)

        return cs

    def write(
        self,
        dest: Path,
        fmt: str = DefaultConfig["structure"]["fmt"],
        *,
        write_children: bool = False,
    ) -> Path:
        """
        Write this object into a directory. It will create the following files:

        #. meta.tsv
        #. sequence.tsv
        #. structure.fmt

        Existing files will be overwritten.

        :param dest: A writable dir to save files to.
        :param fmt: Structure format to use. Supported formats are "pdb", "cif",
            and "mmtf". Adding ".gz" (eg, "mmtf.gz") will lead to gzip
            compression.
        :param write_children: Recursively write :attr:`children`.
        :return: Path to the directory where the files are written.
        """

        if self.is_empty:
            raise MissingData("Attempting to write an empty chain structure")

        dest.mkdir(exist_ok=True, parents=True)

        fnames = DefaultConfig["filenames"]

        self.seq.write(dest)
        self.structure.write(dest / f"{fnames['structure_base_name']}.{fmt}")
        if self.variables:
            self.variables.write(dest / fnames["variables"])

        if write_children:
            for child in self.children:
                child_dir = dest / fnames["segments_dir"] / child.id
                child.write(child_dir, fmt, write_children=True)

        return dest

    def summary(
        self, meta: bool = True, children: bool = False, ligands: bool = False
    ) -> pd.DataFrame:
        s = self.seq.summary(meta=meta, children=False)
        s[DefaultConfig["colnames"]["id"]] = [self.id]
        parent_id = np.NAN if self.parent is None else self.parent.id
        s[DefaultConfig["colnames"]["parent_id"]] = [parent_id]
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
    skip_if_match: str = DefaultConfig["mapnames"]["seq1"],
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
        pos1 = reference.seq[DefaultConfig["mapnames"]["enum"]]
        pos_pairs = zip(pos1, pos2, strict=True)
    else:
        pos_pairs = zip(
            reference.seq[DefaultConfig["mapnames"]["enum"]],
            reference.seq[map_name],
            strict=True,
        )

    pos_pairs = filter(lambda x: x[0] is not None and x[1] is not None, pos_pairs)
    _pos1, _pos2 = unzip(pos_pairs)
    _pos1, _pos2 = map(list, [_pos1, _pos2])

    ref_new = ChainStructure(
        reference.structure.extract_positions(_pos1), reference.chain_id
    )
    c_new = ChainStructure(c.structure.extract_positions(_pos2), c.chain_id)

    return ref_new, c_new


if __name__ == "__main__":
    raise RuntimeError
