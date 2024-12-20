from __future__ import annotations

import logging
import typing as t
from collections import abc
from io import TextIOBase
from pathlib import Path

import numpy as np
import pandas as pd

from lXtractor.chain import ChainSequence, ChainStructure
from lXtractor.chain.base import topo_iter
from lXtractor.chain.list import ChainList, _wrap_children
from lXtractor.core.base import SeqReader, ApplyT, FilterT
from lXtractor.core.config import DefaultConfig
from lXtractor.core.exceptions import (
    AmbiguousMapping,
    MissingData,
    NoOverlap,
    FormatError,
    InitError,
)
from lXtractor.util.seq import read_fasta

LOGGER = logging.getLogger(__name__)

__all__ = ("Chain",)


# TODO: support for empty chain in methods


class Chain:
    """
    A container, encompassing a :class:`ChainSequence` and possibly many
    :class:`ChainStructure`'s corresponding to a single protein chain.

    A typical use case is when one wants to benefit from the connection
    of structural and sequential data, e.g., using single full canonical
    sequence as :attr:`_seq` and all the associated structures within
    :attr:`structures`. In this case, this data structure makes it easier
    to extract, annotate, and calculate variables using canonical sequence
    mapped to the sequence of a structure.

    Typical workflow:

        #. Initialize from some canonical sequence.
        #. Add structures and map their sequences.
        #. ???
        #. Do something useful, like calculate variables using canonical
            sequence's positions.

    .. code-block:: python

        c = Chain.from_sequence((header, _seq))
        for s in structures:
            c.add_structure(s)

    """

    __slots__ = ("_seq", "_id", "structures", "_parent", "children")

    def __init__(
        self,
        seq: ChainSequence,
        structures: abc.Iterable[ChainStructure] | None = None,
        parent: Chain | None = None,
        children: abc.Iterable[Chain] | None = None,
    ):
        """

        :param seq: A chain sequence.
        :param structures: Chain structures corresponding to a single protein
            chain specified by `_seq`.
        :param parent: A parent chain this chain had descended from.
        :param children: A collection of children.
        """
        self._seq: ChainSequence = seq

        #: A collection of structures corresponding to :attr:`_seq`.
        if structures is None:
            structures = ChainList([])
        else:
            if not isinstance(structures, ChainList):
                structures = ChainList(structures)
        self.structures: ChainList[ChainStructure] = structures

        #: A parent chain this chain had descended from.
        self._parent: Chain | None = parent

        #: A collection of children preferably obtained using
        #: :meth:`spawn_child`.
        self.children: ChainList[Chain] = _wrap_children(children)

        self._id = self._make_id()

        for c in self.children:
            c.parent = self

    def __eq__(self, other: t.Any) -> bool:
        if isinstance(other, Chain):
            return (
                self.id == other.id
                and self.seq == other.seq
                and self.structures == other.structures
            )
        return False

    def __hash__(self) -> int:
        return hash(self.id)

    def _make_id(self) -> str:
        parent = "" if self.parent is None else f"<-({self.parent.id})"
        return f"Chain({self.seq.id_strip_parents()}){parent}"

    @property
    def id(self) -> str:
        """
        :return: Chain identifier derived from its :attr:`_seq` ID.
        """
        return self._id

    @property
    def seq(self) -> ChainSequence:
        return self._seq

    @seq.setter
    def seq(self, value: ChainSequence) -> None:
        if not isinstance(value, ChainSequence):
            raise TypeError(f"Invalid type {type(value)}")
        self._seq = value
        self._id = self._make_id()

    @property
    def parent(self) -> t.Self | None:
        return self._parent

    @parent.setter
    def parent(self, value: t.Self | None):
        if not isinstance(value, (type(self), type(None))):
            raise TypeError(
                f"Invalid type {type(value)}. "
                f"Parent must be of the same type {type(self)}"
            )
        self._parent = value
        self._id = self._make_id()

    @property
    def meta(self) -> dict[str, str]:
        """
        :return: A :meth:`seq`'s :attr:`ChainSequence.meta`.
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
        :return: A list of categories from :attr:`_seq`'s
            :attr:`ChainSequence.meta`.
        """
        return self.seq.categories

    def __repr__(self) -> str:
        return self.id

    def __str__(self) -> str:
        return self.id

    def iter_children(self) -> abc.Generator[list[Chain], None, None]:
        """
        Iterate :attr:`children` in topological order.

        See :meth:`ChainSequence.iter_children` and :func:`topo_iter`.

        :return: Iterator over levels of a child tree.
        """
        return topo_iter(self, lambda x: x.children)

    def filter_children(self, pred: FilterT[Chain], inplace: bool = False) -> t.Self:
        """
        Filter children using some predicate.

        :param pred: Some callable accepting chain and returning bool.
        :param inplace: Filter :attr:`children` in place. Otherwise, return
            a copy with only children transformed.
        :return: A chain with filtered children.
        """
        children = self.children.filter(pred)

        if inplace:
            self.children = children
            return self

        return self.__class__(
            seq=self.seq,
            structures=self.structures,
            children=children,
            parent=self.parent,
        )

    def apply_children(self, fn: ApplyT[Chain], inplace: bool = False) -> t.Self:
        """
        Apply some function to children.

        :param fn: A callable accepting and returning the chain type instance.
        :param inplace: Apply to children in place. Otherwise, return a copy
            with only children transformed.
        :return: A chain with transformed children.
        """
        children = self.children.apply(fn)

        if inplace:
            self.children = children
            return self

        return self.__class__(
            seq=self.seq,
            structures=self.structures,
            children=children,
            parent=self.parent,
        )

    def filter_structures(
        self, pred: FilterT[ChainStructure], inplace: bool = False
    ) -> t.Self:
        """
        Filter chain :attr:`structures`.

        :param pred: A callable accepting a chain structure and returning bool.
        :param inplace: Filter :attr:`structures` in place. Otherwise, return
            a copy with only children transformed.
        :return: A chain with filtered structures.
        """
        structures = self.structures.filter(pred)
        if inplace:
            self.structures = structures
            return self
        return self.__class__(
            seq=self.seq,
            structures=structures,
            children=self.children,
            parent=self.parent,
        )

    def apply_structures(
        self, fn: ApplyT[ChainStructure], inplace: bool = False
    ) -> t.Self:
        """
        Apply some function to :attr:`structures`.

        :param fn: A callable accepting and returning a chain structure.
        :param inplace: Apply to :attr:`structures` in place. Otherwise, return
            a copy with only children transformed.
        :return: A chain with transformed structures.
        """
        structures = self.structures.apply(fn)
        if inplace:
            self.structures = structures
            return self
        return self.__class__(self.seq, structures, self.parent, self.children)

    @classmethod
    @t.overload
    def from_seq(
        cls, inp: Path | TextIOBase, read_method: SeqReader = read_fasta
    ) -> ChainList[t.Self]:
        ...

    @classmethod
    @t.overload
    def from_seq(
        cls, inp: str | ChainSequence, read_method: SeqReader = read_fasta
    ) -> t.Self:
        ...

    @classmethod
    @t.overload
    def from_seq(
        cls,
        inp: abc.Iterable[str] | tuple[str, str],
        read_method: SeqReader = read_fasta,
    ) -> t.Self | ChainList[t.Self]:
        ...

    @classmethod
    def from_seq(
        cls,
        inp: str
        | tuple[str, str]
        | ChainSequence
        | Path
        | TextIOBase
        | abc.Iterable[str],
        read_method: SeqReader = read_fasta,
    ) -> t.Self | ChainList[t.Self]:
        # TODO: consider removing and passing the functionality to init
        """

        :param inp: A string of with a sequence or a pair (header, _seq).
            Otherwise, something that the `read_method` accepts.
        :param read_method: A callable accepting a path to a file or opened
            file or an iterable over the file lines and returning pairs
            (header, _seq).
        :return: If a single sequence is provided as a string or pair,
            return an initialized chain. Otherwise, use `read_method` to parse
            the input and embed the resulting :class:`Chain`'s into
            a :class:`ChainList`.
        """

        match inp:
            case str():
                return cls(ChainSequence.from_string(inp))
            case ChainSequence():
                return cls(inp)
            case [str(), str()]:
                return cls(ChainSequence.from_string(inp[1], name=inp[0]))
            case _:
                return ChainList(
                    cls(ChainSequence.from_string(seq, name=name))
                    for name, seq in read_method(inp)
                )

    @classmethod
    def read(
        cls,
        path: Path,
        *,
        search_children: bool = False,
    ) -> Chain:
        """
        :param path: A path to a directory with at least sequence and
            metadata files.
        :param search_children: Recursively search for child segments and
            populate :attr:`children`.
        :return: An initialized chain.
        """
        fnames = DefaultConfig["filenames"]
        seq = ChainSequence.read(path, search_children=False)

        structures = [
            ChainStructure.read(p) for p in (path / fnames["structures_dir"]).glob("*")
        ]
        c = Chain(seq, structures)
        if search_children:
            for child_path in (path / fnames["segments_dir"]).glob("*"):
                child = Chain.read(child_path, search_children=True)
                child.parent = c
                c.children.append(child)
        return c

    @classmethod
    def make_empty(cls) -> t.Self:
        return cls(ChainSequence.make_empty())

    def write(
        self,
        dest: Path,
        *,
        str_fmt: str = DefaultConfig["structure"]["fmt"],
        write_children: bool = True,
    ) -> Path:
        """
        Create a disk dump of this chain data.
        Created dumps can be reinitialized via :meth:`read`.

        :param dest: A writable dir to hold the data.
        :param str_fmt: A format to write :attr:`structures` in.
        :param write_children: Recursively write :attr:`children`.
        :return: Path to the directory where the files are written.
        """
        dest.mkdir(parents=True, exist_ok=True)

        fnames = DefaultConfig["filenames"]
        self.seq.write(dest, write_children=False)

        if self.structures:
            str_dir = dest / fnames["structures_dir"]
            str_dir.mkdir(exist_ok=True)
            for s in self.structures:
                s.write(str_dir / s.id, str_fmt, write_children=False)

        for c in self.children:
            c.write(
                dest / fnames["segments_dir"] / c.id,
                str_fmt=str_fmt,
                write_children=write_children,
            )
        return dest

    def add_structure(
        self,
        structure: ChainStructure,
        *,
        check_ids: bool = True,
        map_to_seq: bool = True,
        map_name: str = DefaultConfig["mapnames"]["map_canonical"],
        add_to_children: bool = False,
        **kwargs,
    ):
        """
        Add a structure to :attr:`structures`.

        :param structure: A structure of a single chain corresponding to
            :attr:`_seq`.
        :param check_ids: Check that existing :attr:`structures`
            don't encompass the structure with the same :meth:`id`.
        :param map_to_seq: Align the structure sequence to the :attr:`_seq` and
            create a mapping within the former.
        :param map_name: If `map_to_seq` is ``True``, use this map name.
        :param add_to_children: If ``True``, will recursively add structure to
            existing children according to their boundaries mapped to the
            structure's numbering. Consequently, this requires mapping, i.e.,
            ``map_to_seq=True``.
        :param kwargs: Passed to :meth:`ChainSequence.map_numbering`.
        :return: Mutates :attr:`structures` and returns nothing.
        :raise ValueError: If `check_ids` is ``True`` and the structure
            id clashes with the existing ones.
        """
        if check_ids:
            ids = [s.id for s in self.structures]
            if structure.id in ids:
                raise ValueError(
                    f"Protein already contains structure {structure.id}. "
                    f"Remove it first or disable `check_ids`"
                )
        if map_to_seq:
            structure.seq.map_numbering(self.seq, name=map_name, **kwargs)
        self.structures.append(structure)
        if add_to_children and len(self.children) > 0:
            for c in self.children:
                sub = structure.spawn_child(
                    c.seq.start,
                    c.seq.end,
                    c.name,
                    map_from=map_name,
                )
                c.add_structure(sub, map_to_seq=False, add_to_children=True)

    def transfer_seq_mapping(
        self,
        map_name: str,
        link_map: str = DefaultConfig["mapnames"]["map_canonical"],
        link_map_points_to: str = "i",
        **kwargs,
    ):
        """
        Transfer sequence mapping to each :attr:`ChainStructure._seq` within
        :attr:`structures`.

        This method simply utilizes :meth:`ChainSequence.relate` to transfer
        some map from the :attr:`_seq` to each :attr:`ChainStructure._seq`.
        Check :meth:`ChainSequence.relate` for an explanation.

        :param map_name: The name of the map to transfer.
        :param link_map: A name of the map existing within
            :attr:`ChainStructure._seq` of each structure in :attr:`structures`.
        :param link_map_points_to: Which sequence values of the `link_map`
            point to.
        :param kwargs: Passed to :meth:`ChainSequence.relate`
        :return: Nothing.
        """
        for s in self.structures:
            self.seq.relate(s.seq, map_name, link_map, link_map_points_to, **kwargs)

    def generate_patched_seqs(
        self,
        numbering: str = DefaultConfig["mapnames"]["enum"],
        link_name: str = DefaultConfig["mapnames"]["map_canonical"],
        link_points_to: str = "i",
        **kwargs,
    ) -> abc.Generator[ChainSequence, None, None]:
        """
        Generate patched sequences from chain structure sequences.

        For explanation of the patching process see
        :meth:`lXtractor.chain.sequence.ChainSequence.patch`.

        :param numbering: Map name referring to a numbering scheme to infer
            gaps from.
        :param link_name: Map name linking structure sequence to the canonical
            sequence.
        :param link_points_to: Map name in the canonical sequence that
            `link_name` refers to.
        :param kwargs: Passed to
            :meth:`lXtractor.chain.sequence.ChainSequence.patch`.
        :return: A generator over patched structure sequences.
        """
        for s in self.structures.sequences:
            yield self.seq.patch(s, numbering, link_name, link_points_to, **kwargs)

    def spawn_child(
        self,
        start: int,
        end: int,
        name: str | None = None,
        category: str | None = None,
        *,
        subset_structures: bool = True,
        tolerate_failure: bool = False,
        silent: bool = False,
        keep: bool = True,
        seq_deep_copy: bool = False,
        seq_map_from: str | None = None,
        seq_map_closest: bool = True,
        seq_keep_child: bool = False,
        str_deep_copy: bool = False,
        str_map_from: str | None = None,
        str_map_closest: bool = True,
        str_keep_child: bool = True,
        str_seq_keep_child: bool = False,
        str_min_size: int | float = 1,
        str_accept_fn: abc.Callable[[ChainStructure], bool] = lambda _: True,
    ) -> t.Self:
        """
        Subset a :attr:`_seq` and (optionally) each structure in
        :attr:`structures` using the provided :attr:`_seq` boundaries
        (inclusive).

        :param start: Start coordinate.
        :param end: End coordinate.
        :param name: Name of a new chain.
        :param category: Spawned child category. Any meaningful tag string that
            could be used later to group similar children.
        :param subset_structures: If ``True``, subset each structure in
            :attr:`structures`. If ``False``, structures are not inherited.
        :param tolerate_failure: If ``True``, a failure to subset a structure
            doesn't raise an error.
        :param silent: Supress warnings for errors when `tolerate_failure` is
            ``True``.
        :param keep: Save created child to :attr:`children`.
        :param seq_deep_copy: Deep copy potentially mutable sequences within
            :attr:`_seq`.
        :param seq_map_from: Use this map to obtain coordinates within
            :attr:`_seq`.
        :param seq_map_closest: Map to the closest matching coordinates of
            a :attr:`_seq`. See :meth:`ChainSequence.map_boundaries`
            and :meth:`ChainSequence.find_closest`.
        :param seq_keep_child: Keep a spawned :class:`ChainSequence` as a child
            within :attr:`_seq`. Should be ``False`` if `keep` is ``True`` to
            avoid data duplication.
        :param str_deep_copy: Deep copy each sub-structure.
        :param str_map_from: Use this map to obtain coordinates within
            :attr:`ChainStructure._seq` of each structure.
        :param str_map_closest: Map to the closest matching coordinates of
            a :attr:`_seq`. See :meth:`ChainSequence.map_boundaries`
            and :meth:`ChainSequence.find_closest`.
        :param str_keep_child: Keep a spawned sub-structure as a child in
            :attr:`ChainStructure.children`. Should be ``False`` if `keep` is
            ``True`` to avoid data duplication.
        :param str_seq_keep_child: Keep a sub-sequence of a spawned structure
            within the :attr:`ChainSequence.children` of
            :attr:`ChainStructure._seq` of a spawned structure. Should be
            ``False`` if `keep` or `str_keep_child` is ``True`` to avoid
            data duplication.
        :param str_min_size: A minimum number of residues in a structure to be
            accepted after subsetting.
        :param str_accept_fn: A filter function accepting a
            :class:`~lXtractor.chain.structure.ChainStructure` and returning
            a boolean value indicating whether this structure should be
            retained in :attr:`structures`.
        :return: A sub-chain with sub-sequence and (optionally) sub-structures.
        """

        def subset_structure(structure: ChainStructure) -> ChainStructure | None:
            try:
                c = structure.spawn_child(
                    start,
                    end,
                    name,
                    category,
                    map_from=str_map_from,
                    map_closest=str_map_closest,
                    deep_copy=str_deep_copy,
                    keep=str_keep_child,
                    keep_seq_child=str_seq_keep_child,
                )
                if len(c) >= str_min_size and str_accept_fn(c):
                    return c
                structure.children.remove(c)
                return None
            except (
                AmbiguousMapping,
                MissingData,
                NoOverlap,
                FormatError,
                InitError,
            ) as e:
                msg = (
                    f"Cannot spawn child structure from {structure} "
                    f"using boundaries {start, end}."
                )
                if tolerate_failure:
                    if not silent:
                        LOGGER.warning(msg)
                else:
                    raise InitError(msg) from e
                return None

        name = name or self.seq.name

        seq = self.seq.spawn_child(
            start,
            end,
            name,
            category,
            map_from=seq_map_from,
            map_closest=seq_map_closest,
            deep_copy=seq_deep_copy,
            keep=seq_keep_child,
        )

        structures = None
        if subset_structures:
            structures = [
                s for s in map(subset_structure, self.structures) if s is not None
            ]

        child = Chain(seq, structures, self)
        if keep:
            self.children.append(child)
        return child

    def summary(
        self, meta: bool = True, children: bool = False, structures: bool = True
    ) -> pd.DataFrame:
        s = self.seq.summary(meta=meta, children=False)
        s[DefaultConfig["colnames"]["id"]] = [self.id]
        parent_id = np.nan if self.parent is None else self.parent.id
        s[DefaultConfig["colnames"]["parent_id"]] = [parent_id]
        if structures and self.structures:
            str_summaries = pd.concat(
                s.summary(meta=meta, children=False) for s in self.structures
            )
            str_summaries["Structure"] = True
            str_summaries["ParentChain"] = self.id
            s = pd.concat([s, str_summaries])
        if children and self.children:
            child_summaries = pd.concat(
                c.summary(meta=meta, children=children) for c in self.children
            )
            s = pd.concat([s, child_summaries])
        return s


if __name__ == "__main__":
    raise RuntimeError
